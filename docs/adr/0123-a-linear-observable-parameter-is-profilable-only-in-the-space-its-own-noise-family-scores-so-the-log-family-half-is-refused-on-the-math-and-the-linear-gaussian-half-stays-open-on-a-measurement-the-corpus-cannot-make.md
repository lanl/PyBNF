# A linear observable parameter is profilable only in the space its own noise family scores, so the log-family half is refused on the math and the linear-Gaussian half stays open on a measurement the corpus cannot make (issue #572)

**Status: Narrowed, not decided (2026-08-21).** ADR-0066/0099 profile a declared column's
optimal multiplicative **scale** out analytically, and ADR-0108 (#562) profiles out an
estimated **noise scale**. #572 proposed the third member — an additive **offset**, and the
coupled `(scale, offset)` pair — by variable projection, and asked for an evaluation before
anything was built.

The evaluation settles two things and leaves the main question open:

* **Settled, on the mathematics.** Variable projection needs the family's residual to be
  `d - y` with `y` affine in the profiled coefficients. PyBNF's **log** families
  (`lognormal`, `lnnormal`) score `log d - log(a*s + b)`, which is affine in `log a` only when
  `b == 0` and in `b` never. The coupled form has no closed form there and must be **refused**,
  not approximated. The pure-scale form survives as ADR-0066's geometric-mean ratio.
* **Settled, on the interface.** A parameter that any noise source also reads is not a free
  linear coefficient and must be refused **by resolved name**, formula sources included —
  because the obvious rule misses one of the two real instances.
* **Open.** Whether profiling is worth having on a **linear-scale** family — the case #572 is
  actually about — is *not* answered here, and this ADR is explicit that the evaluation did not
  answer it. The two measurements that would are named at the end.

No code changes.

## This ADR replaces a wrong conclusion

The first version of this ADR concluded "not adopted", on two arguments that do not hold:

1. *"The corpus has no instances."* PyBNF is general-purpose. Our 23-slug corpus is PEtab
   imports, which skew `lognormal`; a conf written by hand uses `sos` or `chi_sq`, which is a
   **linear** scale — the case where the construction works. `Smith_BMCSystBiol2013`, the one
   `objective = sos` slug, carries nine profilable scales. If anything the sample under-counts
   the feature's domain relative to PyBNF's native idiom. Instance counts in this corpus are
   evidence about Benchmark-Models-PEtab, not about the merit of a general facility.
2. *"The landscape measurement kills it."* It does not, because of where it was taken. The
   collapse documented below happens **because** the residual is on a log scale — and log
   families are exactly what argument (1) above already excludes. The evaluation landscape-tested
   the method only in the regime where it cannot be used. See "What was not measured".

Both corrections are recorded rather than quietly applied, because the numbers underneath them
are sound and reusable; it was the inference from them that was wrong.

## How it was measured

Two tools, both in the corpus (`Grein-2026-benchmark-subset-I/tools/`): `linear_scope.py`
classifies every declared free parameter reaching an observable (affine? pure multiplicative
factor? what residual space? tied across how many series? also read by a noise source?), and
`linear_profile.py` simulates once per point and then minimizes **PyBNF's own
`evaluate_multiple`** over the linear parameters with θ held fixed.

The profile is taken *numerically* on purpose: it is the exact conditional optimum whatever the
family is, so the landscape question separates from the closed-form question. ADR-0108 pinned
its own σ closed form against a numeric minimization of the reported objective for the same
reason. **Neither tool runs a fit** — a `Borghans` point is one simulation plus a few hundred
re-scorings of it.

Three self-checks, all passed: the independently computed flat-line reference reproduces
`-165.982113` at every point with spread `0` (the attractor already on record for that slug);
the reference optimum and PEtab nominal point re-score to their recorded values within the
σ-profiling and build difference; and the profiled score never exceeded the searched score at
any of the **279** points measured, which it cannot, since the searched value is in the set
being minimized over.

## Finding 1 — the residual's space decides what is profilable, and that is general

`Phi = [s, 1]`, `c* = (Phi^T W Phi)^-1 Phi^T W d` is an identity about a residual `d - Phi c`.
A PyBNF noise family declares the space its residual lives in
(`family.additive_on.ln_base`: `0` linear, `ln 10` for `lognormal`, `1` for `lnnormal`). So:

| family | pure scale `a*s` | offset `s + b` | coupled `a*s + b` |
|---|---|---|---|
| linear (`gaussian`, `laplace`, `sos`) | least squares | least squares | least squares |
| log (`lognormal`, `lnnormal`) | **geometric-mean ratio** (ADR-0066) | none | none |

This is the load-bearing result and it has nothing to do with which models we happen to own.
A correct implementation refuses a log-family observable unless the formula is **homogeneous**
in the parameter, and applies the geometric-mean form rather than least squares when it is.
Note "homogeneous" is a property of the whole formula: `Borghans`'s `scale` in
`Z_state*scale + offset` is *not* a pure scale, because setting it to zero leaves `offset`.

## Finding 2 — the double-binding refusal is needed, and the obvious rule is wrong

#572 asks for `Fiedler_BMCSystBiol2016` to be refused by name and asks whether anything else
hides the pattern. Both check out, and the answer to the second is **yes**:

* **`Fiedler`** — confirmed exactly as described: all eight `s_pErk_*` / `s_pMek_*` tokens are
  bound to `observableParameter1_*` **and** `noiseParameter1_*` in the same per-measurement
  tables.
* **`Raia_CancerResearch2011`** — the same defect by a different route, and absent from the
  issue. Five `scaling_*` parameters are the observable scale *and* appear inside a
  `prediction_formula` σ
  (`CD274mRNA*scaling_CD274mRNA*sd_CD274mRNA_rel + sd_CD274mRNA_abs`), so moving the scale moves
  that observable's σ.

A rule that detects the `noiseParameter` placeholder catches `Fiedler` and **misses `Raia`**.
The test has to be on the resolved parameter *names* every noise source reads, formula sources
included. Thirteen of the corpus's 47 affine observable parameters are double-bound this way —
so this is not a corner, it is a quarter of the population, and it is a design constraint on
any implementation regardless of what is decided below.

## Finding 3 — the one in-domain measurement is positive

`Laske`'s `Int_nuc_off` is the corpus's only additive offset on a **linear-scale** family: a
Gaussian offset tied across nine observables. It is exactly #572's case, minus the coupling
with a scale, and it behaves the way a redundant nuisance parameter should:

* it profiles to ≈ `57.13` at essentially **every** box draw, regardless of θ — i.e. it carries
  no information about the dynamics, which is the whole argument for removing it from a search;
* it compresses the spread over the box by ~20× (IQR `1.01e9 -> 5.01e7`, 20 draws);
* it moves the PEtab nominal point by `0.02`.

One parameter in one slug is thin evidence, and σ is searched rather than profiled there
(`Laske` refuses `noise_profiling`, so the absolute magnitudes are inflated). But it is the only
measurement taken inside the construction's actual domain, and it points **for** the feature,
not against it.

## Finding 4 — the open question is redundancy with ADR-0108, not soundness

`Weber_BMC2015`, five pure Gaussian scales, 20 box draws, measured twice:

| | searched IQR | profiled IQR | rank corr |
|---|---:|---:|---:|
| σ **searched** (as the conf ships) | 1.96e18 | 3.38e5 | 0.43 |
| σ **profiled** (`noise_profiling = 1`) | 1.67e3 | **1.72e3** | **0.87** |

With ADR-0108 switched on, profiling all five observable scales changes the spread of the
landscape by **nothing** — marginally worse — and leaves the ranking almost intact. The
thirteen-orders-of-magnitude improvement in the first row is real and it is the **σ profile's**:
a wrong scale inflates residuals, and a profiled σ absorbs the inflation.

`Brannmark_JBC2010` points the other way, and the reason matters: `noise_profiling = 1` is
**refused** there (`IRS1_P` takes σ from a `PerMeasurementFormulaSigma`), so the scale profile
has the field to itself and does compress the landscape (IQR `1.01e17 -> 3.94e12`, rank corr
0.62).

So the real question this evaluation raises is not "is the construction sound" — on a linear
family it plainly is — but **"how much of it is a re-delivery of `noise_profiling`, and does
that depend on whether `noise_profiling` is available for the slug?"** Two slugs, n = 20 and
n = 25, disagreeing for an explicable reason, is not an answer.

This also disposes of #572's proposed ordering rule. "Linear-first, then σ in closed form"
describes a situation that does not arise on `Brannmark`, the slug where the linear profile
matters most, because σ is not profilable there at all. A linear-profiling switch has to work
stand-alone.

## Finding 5 — the scope, read for what it is

Across all 23 subset-I slugs: **47** affine observable-layer parameters in **9** slugs, against
the 22 in 6 the issue tabulates.

| slug | k | affine obs params | linear-lsq | log-geomean | none | why not |
|---|---:|---:|---:|---:|---:|---|
| `Schwen_PONE2015` | 30 | 10 | 0 | 1 | 9 | log family, not homogeneous |
| `Smith_BMCSystBiol2013` | 25 | 9 | 9 | 0 | 0 | — |
| `Fiedler_BMCSystBiol2016` | 22 | 8 | 0 | 0 | 8 | double-bound as a noise parameter |
| `Raia_CancerResearch2011` | 39 | 5 | 0 | 0 | 5 | double-bound as a noise parameter |
| `Weber_BMC2015` | 36 | 5 | 5 | 0 | 0 | — |
| `Brannmark_JBC2010` | 22 | 4 | 4 | 0 | 0 | — |
| `Borghans_BiophysChem1997` | 23 | 2 | 0 | 0 | 2 | log family, not homogeneous |
| `Elowitz_Nature2000` | 21 | 2 | 0 | 0 | 2 | log family, not homogeneous |
| `Laske_PLOSComputBiol2019` | 13 | 2 | 1 | 0 | 1 | log family, not homogeneous |

Two slugs the issue does not list: **`Smith`** (nine tied scales, 36 % of `k = 25`, all
closed-form — the largest opportunity in the corpus) and **`Raia`** (finding 2). It also
undercounts `Brannmark` (4, not 2 — `k_IRP_1Step` and `k_IRSiP_DosR` are scales too) and cites
`Schwen`'s 42 % without noting that 9 of its 10 have no closed form.

**Read this table as a fact about Benchmark-Models-PEtab, not about the feature.** What it does
support is a *priority* claim about PEtab-imported work, and a warning that #572's motivating
examples — `Borghans`, `Elowitz`, `Schwen` — are all in the refused half.

## What was not measured, and why that matters

**The coupled `(scale, offset)` pair on a linear-scale family was never tested, because the
corpus has no instance of it.** That is the case #572 calls "not just one more scalar", and
this evaluation has nothing to say about it. Absence from a 23-problem sample is not evidence
against it.

**Item 3 (head-to-head fits at matched budget) was not run.** It cannot be, without first
building the feature that #572 exists to gate.

**`Smith`'s box landscape was not measured** — 71 s per simulation, and a 9-parameter inner
solve on top. Only its nominal point was scored (`888141 -> 114222`, a large gain, but its
PEtab nominal `sc_*` are far from optimal so this says little).

### The log-family collapse, demoted to what it actually shows

For completeness, since the numbers are sound and were the previous version's headline: taking
the profile **numerically** on a log family — the obvious "no closed form? do it numerically
then" fallback — destroys the landscape.

`Borghans_BiophysChem1997`, 76 box draws that integrate, σ profiled by ADR-0108, profiling
`scale` and `offset`: the searched objective spans `[-160.88, +339.69]` with IQR `129.5`; the
profiled objective spans `[-167.15, -165.98]` with IQR **`0.081`**. Forty-six of 76 draws land
**exactly** on the no-dynamics score `-165.982113`, 73 of 76 within `1.0` of it, and 39 of 76
profile the scale below `1e-6` — the inner solve discards the dynamics, because on a log
residual a wrong-shaped trajectory fits worse than no trajectory at all. Along a section from
the reference optimum out to a box draw, the profiled objective is bit-identical at six
consecutive points where the searched objective rises monotonically. The reference optimum
(`-248.054331 -> -248.054446`) and the nominal point (`-198.207 -> -198.262`) do not move, which
is the envelope theorem. `Elowitz_Nature2000` reproduces it: IQR `70.21 -> 7.77`, 27/57 exactly
at the flat line, 19/57 discarding the dynamics.

**What this is evidence for:** do not offer a numeric profile as a fallback where the closed
form is absent. **What it is not evidence for:** the linear-Gaussian construction, which has no
such mechanism — there the constant column is genuinely in the span, the projection can only
reduce the residual, and `a = 0` is optimal only in the degenerate case that `s` is orthogonal
to the centred data. `Borghans` is also a landscape already on record as pathological (a
wrong-period oscillator scores ~25 NLL units *worse* than a flat line), which is a further
reason not to generalize from it.

## What would decide it

Two measurements, in this order. Neither needs the full feature.

1. **A synthetic linear-Gaussian fixture carrying a real coupled `(scale, offset)` pair.** The
   corpus cannot supply one; building one is a few lines in the tutorial-fixture style, and it
   puts the question in its own domain on a landscape nobody thinks is pathological. Run
   `linear_profile.py` against it exactly as above. This is the measurement the evaluation is
   missing, and it is cheap.
2. **The ADR-0108 redundancy question, on `Smith` and `Brannmark`** — the two slugs where the
   scale profile is *not* redundant with a profiled σ (`Smith` has no σ at all, `Brannmark`
   refuses `noise_profiling`). Not on `Weber`, where finding 4 shows there is nothing left for
   it to collect. This is where #572's item 3 belongs.

If (1) shows the coupled pair behaves like `Int_nuc_off` did and (2) shows a benefit that
`noise_profiling` is not already delivering, the feature is worth building, scoped to:

* linear-scale families, plus the ADR-0066 geometric-mean form for a homogeneous parameter on a
  log family; every other log-family parameter refused with the reason;
* refusal by resolved parameter name for anything a noise source also reads (finding 2);
* a stacked solve over the series a parameter is tied across — **9** of the corpus's 19
  closed-form-profilable parameters are tied across 2 to 11 experiments (`Smith`'s `sc_GLUT_2B`
  and `sc_PI3K` across 11 each), so the per-series solve #572 writes down is not the right
  object for them;
* `k` still counting a profiled linear parameter, and a Bayesian-sampler refusal, both by the
  same arguments ADR-0108 already makes.

## Consequences

* No behaviour change. No new config key. `noise_profiling` (ADR-0108) and the ADR-0066
  `normalization = ..., scale` chain are unaffected.
* #572 stays open, narrowed to the linear-scale case, with the two measurements above as its
  gate.
* Two reusable tools land in the corpus with their gotchas written down:
  `Grein-2026-benchmark-subset-I/tools/linear_scope.py` and `.../linear_profile.py`.

## Prior art

Variable projection (Golub & Pereyra 1973); hierarchical optimization for ODE models (Loos et
al. 2018); pyPESTO/AMICI, which profile scale, offset and σ analytically by default. All three
are stated over an **additive-Gaussian** residual. That assumption is load-bearing, it is the
one finding 1 turns into a refusal, and ADR-0108's citation of the same prior art should be
read with it.
