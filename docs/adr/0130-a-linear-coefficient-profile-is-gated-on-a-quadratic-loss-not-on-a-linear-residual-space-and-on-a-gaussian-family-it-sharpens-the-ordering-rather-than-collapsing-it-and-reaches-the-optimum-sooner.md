# A linear coefficient profile is gated on a quadratic loss, not on a linear residual space, and on a Gaussian family it sharpens the ordering rather than collapsing it and reaches the optimum sooner (issue #572)

**Status: Gate met; recommended for implementation, not yet built (2026-08-25).** ADR-0123 narrowed
#572 to the linear-scale case and named two measurements as its gate. Both have now been taken. They
come out in favour of the feature, they answer the weighting question #572 said had to be decided
deliberately, and they correct one thing ADR-0123 got wrong. **No code changes.**

The short version:

* **The gate passes.** On the synthetic linear-Gaussian fixture, profiling the coupled
  `(scale, offset)` pair does not delete the landscape the way it does on a log family, and the
  ordering it produces tracks the true rate constants where the searched ordering does not.
* **A fit gets there sooner.** At a matched simulation budget the profiled search reaches the
  optimum in about half the simulations with the noise scale profiled, and in well under half
  without. That is #572's item 3, answered where the truth is known.
* **The flat line marks hopeless draws rather than compressing good ones.** How often the profile
  *is* the flat line is a property of how much of the declared box is hopeless, not of the method:
  narrowing the two rate boxes from six decades to two takes it from 18 of 81 draws to 3 of 81, and
  takes the ordering's correlation with the truth from `+0.199` to `+0.771`.
* **ADR-0123's family table is on the wrong axis, and PyBNF already has the right one.** The table
  sorts families by the space their residual lives in, which puts `laplace` in the least-squares
  column beside `gaussian`. Both have `additive_on.ln_base == 0`, so that axis cannot separate
  them. `LikelihoodObjective.is_linear_gaussian()` can, and does. Measured: under `gaussian` the
  least-squares coefficients win by `0.27`, under `laplace` the least-absolute-deviations ones win
  by `0.42`. The gate is a **quadratic loss**.
* **The box question is real and has three answers, not one.** Left inert, as ADR-0108 leaves a
  profiled noise scale, the closed form returns a **negative scale** at 27 of 81 draws for a
  parameter the user declared `loguniform`, and scores a median `14.5` objective units better for
  doing it. Clamp, refuse, or accept the sign flip: an implementation has to choose on purpose.

## Where the measurements and the tools live

In `wshlavacek/BNGL-Models`. The two tools ADR-0123 describes, and the fixture, landed there as
that repository's PR #47; the additions below are PR #48. Every failure mode these tools hide is
written up in `Grein-2026-benchmark-subset-I/tools/README.md`, which is the file to read before
believing any number taken with them.

What was already there: `linear_scope.py` (which free parameters enter an observable linearly, and
in what space), `linear_profile.py` (simulate a box point once, then minimize **PyBNF's own
`evaluate_multiple`** over the linear coefficients), and `Synthetic-2026-linear-observable`, the
fixture ADR-0123 asked for -- `A -> B -> 0` read out as `y = scale * B + offset` under `gaussian`
noise, truth `k1 = 0.8`, `k2 = 0.25`, `scale = 3.0`, `offset = 1.5`, `sigma = 0.05`, 27 points.

What this ADR adds:

* a **`--truth` ranking report** on `linear_profile.py`. The spread of a landscape is not the
  question; whether its ordering tracks the answer is, and nothing was measuring that;
* a **coordinate prescan past two coefficients**. The existing prescan is a full mesh, which is
  `grid ** n` evaluations and unreachable at `Smith_BMCSystBiol2013`'s nine. Sweeping one
  coefficient at a time still walks the whole declared box in each direction, which is the property
  that matters;
* a **no-dynamics reference for a slug with no intercept**. A set of pure scales cannot reach an
  arbitrary constant but can reach zero, and that is its no-dynamics prediction. Without this the
  counter-hypothesis is unmeasurable on most of the corpus;
* **`linear_race.py`**, the head-to-head at a matched simulation budget;
* **`linear_observable_laplace.conf`** and `check_loss_gate.py`, which decide the family gate by
  measurement;
* **`linear_observable_narrow.conf`**, which decides what the flat-line hits mean.

Self-checks, all passed: the profiled score never exceeded the searched score at any point measured,
which it cannot, since the searched value is in the set being minimized over; the flat-line reference
is constant to `2.2e-16` under `--noise-profiling` and visibly is not without it, which is what the
tool warns about; and the profiled coefficients at the truth point come back at `scale = 3.019`,
`offset = 1.501` against a generating `3.0` / `1.5`.

## Finding 1 -- the ordering tracks the truth, which is the question nobody had asked

ADR-0123's own fixture run established that the landscape does not collapse the way `Borghans` does:
the profiled objective spans `62.2` objective units against `Borghans`'s `0.081`, 18 of 81 draws sit
on the flat line against 46 of 76, and the truth stays 62 units below it. That reproduces exactly.

What it did not establish is whether the surviving ordering is any *good*. The searched objective at
a random draw is dominated by how wrong that draw's scale and offset happen to be, so a landscape can
be wide and still rank draws at random. Ranking each landscape against distance to the true
`(k1, k2)`, sigma profiled, 80 draws:

| | searched | profiled |
|---|---:|---:|
| rank correlation with distance to the truth | `-0.229` | `+0.199` |
| distance of the best-ranked draw | `2.779` | `0.704` |
| median distance of the best five | `2.779` | `1.058` |
| draws beating the flat line | 1 of 81 | 79 of 81 |
| score at the truth point | `-63.2612` | `-63.4139` |

Distance is in decades of combined error in `(k1, k2)` against a box median of `2.278`, **with the
fixture's own label swap folded in**. That fold is not cosmetic. Swapping `k1` and `k2` multiplies
`B(t)` by `k2/k1` and leaves its shape alone, so a free scale absorbs the difference exactly and the
two points are the *same fit*; the mirror answer is only ever reached by the side that has the scale
at its optimum, so measured without folding it reads as the profiled side getting further from the
truth as it gets better. The fixture's README has always said the fixture has two global optima; the
tooling now knows.

The searched ordering is worse than useless here -- `-0.229` means it mildly prefers draws that are
further away. The profiled ordering is positive but weak. Finding 2 explains why it is only weak,
and it is not the method.

## Finding 2 -- the flat-line hits measure the box, not the profile

Same fixture, same everything, with the two rate boxes narrowed from `1e-3 .. 1e3` to
`1e-1 .. 1e1` (`linear_observable_narrow.conf`), sigma profiled, 80 draws:

| | six decades per rate | two decades per rate |
|---|---:|---:|
| draws whose profile **is** the flat line | 18 of 81 | **3 of 81** |
| draws beating the flat line | 79 of 81 | **81 of 81** |
| rank correlation with the truth, profiled | `+0.199` | **`+0.771`** |
| rank correlation with the truth, searched | `-0.229` | `-0.094` |
| distance of the best-ranked draw, profiled | `0.704` | `0.170` |
| box median distance | `2.278` | `0.750` |

Over six decades of rate constant most draws produce a trajectory that is flat or instantaneous, and
for those the flat line genuinely **is** the best the observation model can do. Reporting that is the
profile being right, not the profile failing. Narrow the box to where candidates are actual
candidates and the flat-line hits nearly vanish and the ordering becomes strong.

This is the answer to #572's counter-hypothesis, and it is a sharper answer than "the landscape does
not collapse". The worry was that the flat line becomes a floor every candidate sinks to. It is a
**ceiling that hopeless candidates are correctly pinned against**, and it leaves the ordering of the
rest alone.

## Finding 3 -- a fit gets to the optimum on a fraction of the budget

`linear_race.py` runs the same rand/1/bin differential evolution on both sides, from the same six
seeds, at a matched number of **simulations** -- the honest currency, because #572's proposal is to
replace the inner minimization with one linear solve. Here the inner solve is that closed form
(`--closed-form`), so the two sides also cost the same wall clock. The optimum is `-64.8803`.

Sigma profiled, so the searched side carries four parameters and the profiled side two:

| simulations | searched best / median | profiled best / median | searched distance | profiled distance |
|---:|---:|---:|---:|---:|
| 60 | `-1.54` / `-1.09` | `-50.17` / `-37.17` | `3.255` | `0.354` |
| 120 | `-8.26` / `-2.85` | `-64.28` / `-55.81` | `1.735` | `0.328` |
| 250 | `-26.37` / `-16.47` | `-64.73` / `-64.18` | `0.479` | `0.126` |
| 500 | `-55.47` / `-41.42` | `-64.880` / `-64.880` | `0.225` | `0.091` |
| 1000 | `-64.87` / `-64.02` | `-64.880` / `-64.880` | `0.085` | `0.091` |

Sigma searched, so five parameters against three:

| simulations | searched best / median | profiled best / median | searched distance | profiled distance |
|---:|---:|---:|---:|---:|
| 60 | `+7.37` / `+17.32` | `-23.06` / `-19.62` | `2.770` | `1.463` |
| 120 | `-0.51` / `+5.21` | `-40.64` / `-26.47` | `2.600` | `1.584` |
| 250 | `-13.55` / `-2.38` | `-52.75` / `-42.14` | `1.142` | `0.487` |
| 500 | `-16.44` / `-13.46` | `-64.72` / `-62.18` | `0.672` | `0.117` |
| 1000 | `-44.61` / `-36.91` | `-64.880` / `-64.876` | `0.319` | `0.092` |

The profiled side wins at every budget in both tables. With sigma profiled it is at the optimum by
500 simulations where the searched side needs 1000. With sigma searched it is there by 500 to 1000
while the searched side has still not arrived at 1000, its median sitting 28 objective units short.

**This settles the redundancy question ADR-0123 raised.** Removing the linear coefficients pays off
whether or not `noise_profiling` is also on, and it pays off *more* when it is not, because there is
more nuisance left to remove. The landscape ranking alone would have suggested the opposite --
ranking random draws with sigma searched barely improves, `-0.019` to `+0.058` -- and the two are not
in conflict. Ranking random draws is dominated by whichever nuisance is still in the search; running
a search also gets two fewer dimensions to descend, and that helps regardless.

## Finding 4 -- the family gate is the loss, and PyBNF already has the predicate

ADR-0123's finding-1 table reads:

> | family | pure scale | offset | coupled |
> | linear (`gaussian`, `laplace`, `sos`) | least squares | least squares | least squares |

The `laplace` cell is wrong, and the axis is what makes it wrong. Both families have
`additive_on.ln_base == 0`, so a rule read off that attribute -- which is what ADR-0123 invites, and
what `ObjectiveFunction._scale_mode` reads for ADR-0066 -- cannot tell them apart. Variable
projection minimizes a sum of squares; a Laplace likelihood is a sum of **absolute** residuals, so
its conditional optimum over `(scale, offset)` is a least-absolute-deviations fit. The residual being
on a linear scale is necessary and not sufficient.

Measured at the fixture's truth point, same model, same data, same 27 scored points, only the loss
swapped (`check_loss_gate.py`):

| conf | `is_linear_gaussian()` | least squares | by L1 | winner |
|---|---|---:|---:|---|
| `linear_observable.conf` | `True` | **`-62.768`** | `-62.498` | least squares by `0.270` |
| `linear_observable_laplace.conf` | `False` | `-35.370` | **`-35.789`** | L1 by `0.420` |

The coefficients themselves barely move -- `(3.019, 1.501)` against `(3.003, 1.498)` -- which is why
this is easy to miss and expensive to get wrong: the answer looks right and is not the optimum.

**The fix needs no new predicate.** `LikelihoodObjective.is_linear_gaussian()` already exists, added
as the config-time precondition for the DREAM Kalman proposal (ADR-0067 Stage 3), and it returns
`True` and `False` for these two confs respectively. `aligned_prediction_data`, from the same work,
already refuses a non-Gaussian objective outright and already supplies the aligned
`(prediction, observation, variance)` vectors the solve needs. An implementation should read those
two seams rather than `additive_on.ln_base`.

(`laplace`'s *pure-scale* case does have a closed form: minimizing `sum_i w_i |d_i - a s_i|` over `a`
is a weighted median of `d_i / s_i` with weights `w_i |s_i|`. That is a different construction and is
not proposed here.)

Nothing about the corpus census changes. No slug uses `laplace`, and all 20 closed-form-profilable
parameters sit on `gaussian`, `sos` or `chi_sq`. The correction is about a general implementation,
which is the thing PyBNF is.

## Finding 5 -- the closed form is exact, and the box is a decision

With sigma searched, so that `W` is the same matrix on both sides, #572's
`c* = (Phi^T W Phi)^-1 Phi^T W d` was computed at all 81 points and compared against the numerical
conditional optimum:

* **54 of 81 agree to `1.5e-14` relative.** The construction is exactly right wherever both methods
  can reach the same point.
* **27 of 81 do not, and the closed form wins**, by a median `14.5` objective units. There it
  returns a **negative `scale`** -- the model predicting the mirror image of the data -- which the
  numerical profile cannot follow, because `scale` is declared `loguniform` and its sampling space is
  strictly positive.

The closed form is unconstrained by construction: it does not know the parameter has a declared
support. So "profile it out" and "respect its declared bounds" are not automatically compatible, and
an implementation has to pick one of three, in the open:

1. **An inert box**, which is what ADR-0108 does for a profiled noise scale. Then a `loguniform`
   parameter can come back negative and the fit is better for it. Defensible, and surprising enough
   that it belongs in the run's output rather than in a user's discovery.
2. **A box-constrained solve.** Not the unconstrained solve clipped: clamping one coefficient
   changes where the other belongs. An active set -- solve, clamp whichever coefficient left its box,
   re-solve the rest -- is exact for two coefficients and cheap.
3. **A refusal** for a coefficient whose unconstrained optimum leaves its declared support, with the
   reason.

**The ordering question is settled for the common case.** With one Gaussian sigma shared across the
profiled points, profiling it turns the negative log likelihood into `n/2 * log(sum r^2) + const`,
monotone in the residual sum of squares, so the coefficients that minimize it are the same
least-squares ones and the joint optimum is "least squares, then sigma = rms". Nothing has to be
iterated. It stops being true as soon as two observables carry different estimated sigmas, because
then each group's residual carries its own weight; that case needs an alternating solve or a
documented refusal.

## Finding 6 -- the corpus half of the gate

ADR-0123 named `Smith_BMCSystBiol2013` and `Brannmark_JBC2010` as the two slugs where a linear
profile cannot be a re-delivery of `noise_profiling`, because neither can use it: `Smith` fits under
`objective = sos` and has no sigma at all, and `Brannmark` refuses `noise_profiling` because
`IRS1_P` takes its sigma from a `PerMeasurementFormulaSigma`.

Box draws, each simulated once and then profiled over the slug's own closed-form-profilable
parameters. Neither slug has a known truth, so the ordering question a fixture can answer is not
available here; what is available is the spread and the distance to the no-dynamics reference.

| | `Smith_BMCSystBiol2013` | `Brannmark_JBC2010` |
|---|---:|---:|
| profiled parameters | 9 pure scales of 25 free | 4 pure scales of 22 free |
| objective | `sos`, no sigma at all | `chi_sq`, `noise_profiling` refused |
| draws that integrate | 15 of 15 | 25 of 25 |
| searched: min / median / max | `2.50e8` / `1.64e15` / `4.28e17` | `1.92e10` / `1.13e14` / `9.86e21` |
| profiled: min / median / max | `8.80e4` / `2.08e5` / `2.95e5` | `260.8` / `2.42e11` / `2.99e13` |
| IQR | `4.78e16` -> `9.12e4` | `1.01e17` -> `3.94e12` |
| rank correlation, searched vs profiled | `-0.24` | `+0.64` |

Two things to take from this.

**The compression is enormous and it is not collapse.** `Smith`'s searched landscape spans nine
orders of magnitude across the box; profiled, it spans a factor of `3.4`. Read on its own that is
what #572's counter-hypothesis predicts. It is not, and the no-dynamics reference says why: with
nine *pure* scales and a sum-of-squares objective, sending every scale to zero scores `1.54e6`, and
the **worst** of the 15 profiled draws is a fifth of that. Every draw's dynamics are contributing a
large real improvement. What the profile removed is nine orders of magnitude of "how wrong did this
draw's scales happen to be", not the signal — the same thing finding 2 shows directly on the fixture.

**The two orderings are unrelated.** `Smith`'s `-0.24` over 15 points is, at that sample size,
indistinguishable from no correlation; `Brannmark`'s `+0.64` over 25 is moderate. The searched
ordering is not a noisy version of the profiled one. On the fixture, where the truth is known, the
profiled ordering is the one that tracks it.

What the corpus cannot say is whether a factor of `3.4` is enough spread for a global optimizer to
work with on `Smith`. That is the question finding 3 answers on the fixture, and its answer is yes.

`Smith`'s inner solve is the coordinate prescan described above, capped at `--maxiter 150` because
a nine-dimensional simplex on a seventy-second simulation is otherwise minutes per point. The cap
costs about 2 % on the profiled score, which understates the profiled side.

## Finding 7 -- one correction to ADR-0123 that is not about the decision

ADR-0123's finding 3 says `Laske`'s `Int_nuc_off` "profiles to approximately `57.13` at essentially
every box draw ... i.e. it carries no information about the dynamics". Re-run at the same seed with
the same tool, 15 of 20 draws integrate, and of those 9 profile to `57.13`, one to `52.5`, and **five
to effectively zero** (`3.8e-17` to `2.8e-09`). Nine of fifteen is not "essentially every", and the
five that go to zero are the parameter responding to the trajectory rather than ignoring it. The
compression ADR-0123 reports is real and reproduces -- IQR `1.09e9 -> 3.79e7` here against
`1.01e9 -> 5.01e7` there, the difference being the five draws that failed to integrate. The reading
of it is what overstates.

`Brannmark` reproduces exactly and needs no correction: IQR `1.01e17 -> 3.94e12`, rank correlation
`0.637` against ADR-0123's `0.62`.

## What this supports building

The scope ADR-0123 sketched, amended by the findings above. Ordered as an implementation would take
them, and modelled throughout on ADR-0108, which is the same shape of change:

1. **A whole-fit switch** (`linear_profiling = 1`), refused rather than partially applied, exactly as
   `noise_profiling` is. Profiling some of a fit's linear coefficients while searching others changes
   what the searched ones mean.
2. **A plan method on the objective** that classifies each declared free parameter reaching an
   observable and lists a one-line reason for every refusal. The classification is
   `linear_scope.py`'s: affine in the formula, homogeneous or not, what loss the observable's family
   applies, and whether any noise source reads the name.
3. **Refusals, all before the run starts.** A non-quadratic loss, tested with `is_linear_gaussian()`
   rather than `additive_on.ln_base` (finding 4). A log family unless the parameter is homogeneous,
   in which case the ADR-0066 geometric-mean form applies (ADR-0123 finding 1). Any parameter a noise
   source also reads, tested on **resolved names**, formula sources included, which 13 of the
   corpus's 47 need (ADR-0123 finding 2). A Bayesian sampler, by ADR-0108's argument unchanged:
   profiling maximizes a nuisance out where a posterior integrates it.
4. **One stacked solve per group of tied points**, not a per-series solve. Many of the corpus's
   closed-form-profilable parameters are tied across 2 to 11 experiments (`Smith`'s `sc_GLUT_2B` and
   `sc_PI3K` across 11 each). Coefficients that share an observable are solved jointly; the fixture's
   pair is the two-column case and `Smith`'s nine scales are nine independent one-column cases.
5. **A stated answer to the box question** (finding 5), chosen from the three above rather than
   fallen into, and visible in the run's output when a coefficient leaves its declared support.
6. **The weighting question, answered.** #572 offered three options and said the choice should be
   made deliberately. The solve must be the objective's own conditional minimizer over the
   coefficients, `W = diag(w_i / sigma_i^2)`, which is what `aligned_prediction_data` already
   returns. That is what makes the closed form equal the numerical optimum (finding 5) and what makes
   the envelope theorem apply, so the reduced gradient is the partial and ADR-0099's product rule is
   not needed. It is #572's option (a). It leaves the existing `normalization = ..., scale` chain
   (ADR-0066/0099) alone on its own sigma-unweighted criterion; option (c), migrating ADR-0066 to
   match, would change existing `scale` results and is not proposed.
7. **Rank deficiency.** `Phi = [s, 1]` is singular when `s` is constant over the group, which is
   exactly what a global sampler visits. A pseudo-inverse fallback and the
   `_warn_degenerate_profile` treatment, as #572 asks. `linear_profile.py::_varpro` already falls
   back to `pinv` for this reason.
8. **`k` still counts a profiled coefficient**, and the fitted values are reported beside the results
   rather than as coordinates of the best PSet, both by ADR-0108's arguments, unchanged.
9. **Documentation says what finding 2 says.** How often the profile returns the flat line is a
   statement about the declared box, and a box wide enough that most draws are hopeless will produce
   many of them.

## What is still not measured

* **A head-to-head fit on a real slug.** The race in finding 3 is on the fixture, where the truth is
  known and one simulation costs a hundredth of a second. Doing it on `Smith`, whose simulations take
  seventy seconds, needs the feature itself, which is what #572 exists to gate.
* **A coupled pair tied across experiments.** The fixture's pair sits in one experiment. The corpus
  covers ties (`Smith` across 11, `Brannmark` across 7) but only for pure scales, so the stacked
  solve for a *coupled* pair has no measurement behind it.
* **The `Schwen` reparametrization.** `scale*(IR1 + IR1in + offset)` spans the same two columns as
  `(scale, scale*offset)`, so a solve over the span works and mapping back to the user's declared
  names is ill-posed as `scale -> 0`. #572 flags this and it stays open. It is moot for `Schwen`
  itself, which is a log family and refused on ADR-0123 finding 1.
* **Partial separability inside one observable.** `Schwen`'s `observable_Insulin` mixes two linear
  parameters with two nonlinear ones. `linear_scope.py` classifies per parameter rather than per
  observable, so the information is there, but nothing has exercised a mixed solve.

## Consequences

* No behaviour change and no new config key in this ADR. `noise_profiling` (ADR-0108) and the
  ADR-0066 `normalization = ..., scale` chain are untouched.
* ADR-0123 becomes superseded in part. Its census and its refusals stand. Its finding-1 table is
  corrected by finding 4 and its finding-3 reading of `Laske` by finding 7, and its two open
  measurements are closed by findings 1, 2, 3 and 6.
* #572's gate is met. Whether to build is a separate decision; this ADR records what building it
  would mean, and a follow-up issue tracks it.

## Prior art

Variable projection (Golub and Pereyra 1973); hierarchical optimization for ODE models (Loos et al.
2018); pyPESTO and AMICI, which profile scale, offset and sigma analytically by default. All three
are stated over an additive-**Gaussian** residual. That assumption is load-bearing, and finding 4 is
what happens when it is relaxed to "additive" alone.
