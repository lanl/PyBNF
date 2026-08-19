# An only-ever-tighten clamp is a no-regression rule rather than a property of the model, so `sbml_atol` gains an opt-in that lets the derivation loosen — and one that follows the trajectory (issue #557)

**Status: Accepted and implemented (2026-08-18). Extends ADR-0103 and ADR-0105; supersedes neither.**
Both stand exactly as written for the default path, which is byte-identical after this change. What
this adds is a way for a job to say how far its own model's state may set its tolerance.

## The clamp, and what it is for

ADR-0103 derives a scalar `atol` from the model's median nominal species value and clamps it into
`[1e-16, backend default]`; ADR-0105 derives a per-species vector and clamps each entry into
`[that scalar, backend default]`. The upper clamp in both is the same rule — *the derivation may
only ever tighten* — and ADR-0103's own text says why:

> the derivation can only *tighten*, never loosen, so a model whose species are of order one (or
> larger) keeps the backend default bit-for-bit and every existing trajectory is unchanged.

As a no-regression rule that is sound, and it is why the derivation could be applied to every model
without a flag. The problem is the parenthesis. **"Or larger" is doing a great deal of work.** A
model whose species all sit far above one has a real, computable tolerance need, and the derivation
computes it and then discards it.

`Weber_BMC2015` is the case #557 is filed from: seven species at `1.24e+02 .. 4.21e+07`, median
`4.665e+05`, so `rtol * median` is `4.665e-03` and the clamp hands it `1e-08` — **5.7 decades
tighter than its own scale asks for**. ADR-0105's vector cannot rescue it either, and the way it
fails is worth stating precisely: entries clamp into `[scalar_atol, default_atol]`, the scalar has
itself been clamped to `default_atol`, so that interval collapses to a point, the vector is
elementwise the scalar, and `_derive_per_species_atol` correctly returns `None`. **The per-species
mechanism silently declines exactly the models whose own scale asks for something the ceiling will
not give.**

Over the 22 subset-I slugs with a readable nominal state, the clamp binds on **10** — measured
today, not carried forward:

| slug | median y₀ | wants | gets |
|---|---:|---:|---:|
| `Smith_BMCSystBiol2013` | 7.879e+14 | **7.88e+06** | 1e-08 |
| `Perelson_Science1996` | 1.86e+06 | 1.86e-02 | 1e-08 |
| `Weber_BMC2015` | 4.665e+05 | 4.67e-03 | 1e-08 |
| `Rahman_MBS2016` | 1.365e+04 | 1.37e-04 | 1e-08 |
| `Crauste_CellSystems2017` | 4.046e+03 | 4.05e-05 | 1e-08 |
| `Zhao_QuantBiol2020` | 387 | 3.87e-06 | 1e-08 |
| `Okuonghae_ChaosSolitonsFractals2020` | 212 | 2.12e-06 | 1e-08 |
| `Laske_PLOSComputBiol2019` | 150 | 1.50e-06 | 1e-08 |
| `Oliveira_NatCommun2021` | 24.11 | 2.41e-07 | 1e-08 |
| `Raia_CancerResearch2011` | 1.30 | 1.30e-08 | 1e-08 |

The other 12 either tighten (`Giordano_Nature2020`, `Brannmark_JBC2010`, `Bertozzi_PNAS2020`,
`Armistead_CellDeathDis2024`) or sit at median exactly 1, where the clamp is a no-op.

**Nine of those ten integrate their box samples at the clamped value today** — every one probed
below; `Smith_BMCSystBiol2013` was not, because a single 10-point sensitivity probe on it had not
finished in twenty minutes, which is its own remark about a model whose recorded fit took 13 h. That
is the whole argument for an opt-in rather than a default change: the table is the *scope* of the
problem, not a mandate to move nine working results.

## The decision

**`sbml_atol` takes two settings besides a number, and both are statements about how far the
model's own state may set its tolerance rather than replacements for the derivation.**

```
sbml_atol = auto                  # trust the derivation in both directions
sbml_atol = tracking [<decades>]  # auto's vector as a ceiling, followed down the trajectory
```

A **number** is unchanged and remains the documented off-switch: it pins the scalar and switches
every derivation off, `CVodeSStolerances`, ulp for ulp. **Unset** is unchanged in every respect —
same clamps, same vector, same steady-state pairing.

### `auto` lifts one clamp, and only one

```python
scalar_i = max(rtol * median(y), 1e-16)        # was min(that, default_atol)
atol_i   = max(rtol * y_i, scalar)             # was min(that, default_atol)
```

- **The ceiling goes.** That is the whole of what #557 asks for, and it is safe to make optional
  precisely because it is not safe to make default.
- **The floor stays exactly where ADR-0105's measurement put it.** Read literally, "resolve each
  species to `rtol` of its own magnitude" puts `Brannmark_JBC2010`'s `IRp` at `1.76e-17`, and over
  100 points of that model's own fit box that rule killed **91 of 100** simulations against the
  scalar's 39. Nothing about a model living *above* one revisits that measurement, so #557 moves
  one clamp and leaves the other alone.
- **`_DERIVED_ATOL_FLOOR` stays too.** "Both directions" is a natural reading that would take the
  `1e-16` floor with it. It answers a different question — how much resolution the derivation will
  reach for *on its own*, which past `1e-16` is already under double precision's resolution of a
  state of order one — and a model that genuinely lives down there still says so with a number.

What is left is `max(rtol*y_i, rtol*median)` = `rtol * max(y_i, median)`, which is **bngsim's own
`derive_atol` with the model's median supplied as its `floor`**. So the opt-in is not a fourth
PyBNF-specific rule; it is the backend's rule with PyBNF's answer to the one question the backend
leaves open (bngsim's default floor is the smallest strictly positive species, the reading ADR-0105
measured and rejected). `test_the_loosened_vector_is_bngsims_own_derivation` asserts that against
the library, as `test_only_ever_tighten_is_pybnfs_to_apply_not_the_backends` already does for the
clamped default.

### `tracking` is the half ADR-0105 named as out of reach

ADR-0105 closes by naming what it could not do:

> resolving a species *below* the model's own scale, and a tolerance that follows the trajectory —
> the same thing from two directions […] That is lanl/bngsim#213's `CVodeWFtolerances`.

That is now shipped in bngsim, and `sbml_atol = tracking` reaches it. The rule bngsim installs is
`atol_i(y) = clamp(rtol*|y_i|, ceiling_i * 10**-decades, ceiling_i)`, evaluated at the state being
integrated. **The ceiling is `auto`'s vector**, so:

- tracking is never *looser* than `auto`, only deeper, and `tracking 0` is `auto` exactly (to ~1
  ulp — the same error weights through `CVodeWFtolerances` rather than `CVodeSVtolerances`). That
  is a strict-extension property, not a coincidence, and it is why one internal flag governs both
  settings rather than two modes sharing a derivation by accident;
- the ceiling is stated explicitly from the **nominal** state. bngsim's own default ceiling is
  `"auto"`, which re-derives from the model's **live** state at every `run()` — the shape ADR-0105
  ruled out for the vector, because on a fit that moves initial conditions the tolerance becomes a
  function of the search position, putting a step in the objective wherever the derivation crosses
  a rounding boundary. Invisible in the usual way: the objective still looks correct and only the
  search behaves oddly.

`decades` is optional and an unstated depth is left to bngsim rather than copied here — 12 is a
measured property of that mechanism (where its own accuracy stops improving; past it the limit is
roundoff) and belongs to the library that measured it.

### The rest of the decision

- **The steady-state cutoff is stated under tracking too, for ADR-0105's reason.** bngsim resolves
  a `TrackingAtol` to the *Simulator's own* scalar for every scalar-shaped consumer, and
  `steady_state_tol` is one, so left alone a tracking run would converge `time = inf` measurements
  and pre-equilibration phases against `1e-8` rather than against anything the model said.

- **Tracking applies even when the vector declines.** A model whose species share one scale takes
  the scalar call (19 of 23 slugs), and under tracking that scalar becomes a broadcast ceiling
  rather than a refusal. What tracking adds is *within*-species and over time, which is orthogonal
  to whether the species differ from each other — a model that starts uniform and decays is
  precisely where a vector of initial values has nothing to say.

- **`auto` is meaningful without lanl/bngsim#196.** The scalar is the whole of what ADR-0103 had
  and it is what the clamp acted on, so an older bngsim opting in still gets the looser number.
  The capability probe refines this change rather than gating it.

- **`tracking` is refused, not degraded, without lanl/bngsim#213.** Both at config load
  (`PybnfError`, so the run never starts) and in the model constructor (`ModelError`, for a model
  built directly). A tolerance mode that silently did not apply produces a plausible trajectory and
  a wrong conclusion about the model, which is the failure this whole line of work exists to stop
  making.

- **`sbml_atol` still does not take a hand-written vector**, which is #557's second ask. Its second
  clause — relax `_derive_atol_vector`'s upper clamp — is delivered here, and with it
  `CVodeSVtolerances` becomes reachable from a conf on models that could not reach it before, with
  entries the model's own state supplies. The first clause is left open as **#586**: `sbml_atol` is
  one global key applying to every SBML/Antimony model in a fit, so a positional vector has no
  ordering a conf author can see and a species-keyed one has no unambiguous reading across models
  that do not share species. That needs a per-model tolerance record and its own design.

## Verification

### The premise measurement no longer reproduces, and that is worth saying plainly

#557's headline is that at the clamped tolerance `Weber_BMC2015` is unfittable — 6 of 30
sensitivity-applied box points integrating, in 239 s, against 22 at `sbml_atol = 1e-4`. **On
today's stack that is no longer true.** The same probe, same slug, same seed, same 30 points, same
sensitivity request:

| `sbml_atol` | integrated / 30 | wall |
|---|---:|---:|
| unset (the clamped derivation, `1e-08`) | **30** | 7.2 s |
| `1e-08` explicit | 30 | 7.2 s |
| `1e-04` (what the slug ships) | 30 | 6.6 s |
| `auto` (`4.665e-03` + vector) | 30 | 6.2 s |
| `tracking 6` | 30 | 6.8 s |
| `tracking` (depth 12) | 30 | 7.1 s |

The stack moved between #557's measurement (bngsim 0.12.2 at `114d3b3`) and this one (bngsim
0.13.0), and the documented Weber-specific change in between is **lanl/bngsim#305** — a registered
time-discontinuity root that could never be reached, so the run wedged one ulp below `t = 24`.
bngsim's own entry measures that fix on this very slug and reports error-test failures at the
crossing falling from 33–58 to 0–4 with the step count roughly halving. That is consistent with the
`mxstep` exhaustion #557 attributes to the tolerance, but **this ADR has not bisected it** and does
not claim more than the two observations: the failure was real when measured, and it is not present
now.

Two consequences, and they point in different directions:

- **#557's strongest argument for the change is gone.** "This alone would have made Weber fittable
  with no hand-set number" is no longer a statement about the current code, because Weber is
  fittable at the clamped tolerance now.
- **The defect #557 describes is untouched by that.** The derivation still computes `4.665e-03` for
  Weber and still discards it; the vector still declines on exactly the models that span the most
  decades; `sbml_atol` still could not reach either of bngsim's two newer mechanisms. Those are
  properties of PyBNF's code, and none of them was fixed upstream.

So this ships as what it is — a capability with a measured *cost* rather than a rescue — and the
honest summary of the corpus evidence is below rather than in the issue's original numbers.

### What the loosening is worth on the corpus now

Integrator **steps**, not pass/fail, is the instrument that discriminates: `auto` is a claim about
how much work the tolerance is asking for, and on models that all integrate anyway the wall clock on
a small model cannot see it. 20 points from each slug's own fit box (seed 11), with the gradient
sensitivity request applied, counting every CVODE step and error-test failure across every `run()`
the probe drives:

| slug | clamp | clamped steps | `auto` steps | vs | `tracking` steps |
|---|:--:|---:|---:|---:|---:|
| `Perelson_Science1996` | binds | 8 440 | **3 216** | 0.38x | 9 512 |
| `Weber_BMC2015` | binds | 78 641 | **36 154** | 0.46x | 75 650 |
| `Crauste_CellSystems2017` | binds | 8 617 | **4 162** | 0.48x | 13 751 |
| `Rahman_MBS2016` | binds | 6 298 | **3 709** | 0.59x | 5 783 |
| `Okuonghae_ChaosSolitonsFractals2020` | binds | 19 480 | **11 996** | 0.62x | 25 946 |
| `Laske_PLOSComputBiol2019` | binds | 236 629 | **158 543** | 0.67x | 292 014 |
| `Oliveira_NatCommun2021` | binds | 13 639 | 11 164 | 0.82x | 17 037 |
| `Raia_CancerResearch2011` | binds | 116 506 | 115 127 | 0.99x | 272 650 |
| `Giordano_Nature2020` | tightens | 157 728 | **157 728** | **1.00x** | 197 082 |
| `Brannmark_JBC2010` | tightens | 164 564 | 167 700 | 1.02x | 255 575 |

Four things this says, and one of them is uncomfortable.

- **The loosening is worth 1.5x–2.6x of the integrator's work** on the models the clamp binds
  hardest, at no cost in points integrated. That is the value of the change on the current stack:
  not "this model now runs", but "this model was being charged for resolution nobody asked for". The
  two rows near 1.00x are the ones whose median barely clears the ceiling (`Raia` at 1.30,
  `Oliveira` at 24.1), which is the expected gradient.
- **`Giordano_Nature2020` is bit-identical, 157 728 = 157 728, 126 = 126 error-test failures.** It is
  the control the whole design rests on: a model whose derivation *tightens* cannot see the ceiling,
  so lifting it must change nothing. It changes nothing. (`Brannmark` moves 2% because `auto` also
  releases its vector's top end — `X` from `1e-8` to bngsim's `1.0e-07` — which is the one place the
  two ADR-0105 slugs are touched at all, and it costs steps rather than saving them.)
- **Error-test failures often go *up* while steps go down** (`Perelson` 10 → 52, `Zhao` 147 → 1121).
  That is what a looser tolerance looks like from inside: bigger steps, more of them rejected, fewer
  overall. It is also the mechanism behind the smoothness cost below, so it is worth having in view
  rather than reporting the step count alone.
- **`Laske_PLOSComputBiol2019` lost a box point**, 20/20 to 19/20. Re-run at a second seed both arms
  give 19/20 (and the clamped arm is the one that spends 349 717 steps against `auto`'s 242 173), so
  this is not a systematic regression — but it is the direction the opt-in's risk lies in, and it is
  exactly why this is opt-in. `Zhao_QuantBiol2020` is the same shape one step milder: 97 675 steps to
  68 929, with error-test failures up 7.6x and no point lost.

### It is not an accuracy trade

`J_paper` at the PEtab nominal point — the quantity `nominal_check.json` pins, computed through
`likelihood_information_criteria` exactly as an end-of-fit does:

| slug | arm | `J_paper` | `OG_nominal` |
|---|---|---:|---:|
| `Weber_BMC2015` | clamped (`1e-08`) | 296.2018011 | −0.000201 |
| | `1e-04` (what it ships) | 296.2018264 | −0.000176 |
| | `auto` (`4.665e-03`) | 296.2018660 | −0.000136 |
| | `tracking` | 296.2018022 | −0.000200 |
| `Giordano_Nature2020` | clamped | 287.62776135703666 | 3775.9692594667868 |
| | `auto` | **287.62776135703666** | **3775.9692594667868** |

Weber's whole 5.7-decade sweep moves the objective in its **sixth decimal**, and `nominal_check.json`
holds under every arm. Giordano's is identical to all 17 digits, which is the control again.

What a looser tolerance costs is **smoothness**, and the distinction is the reason `Weber_BMC2015`
ships `sbml_atol = 1e-4` rather than the derivation's own `4.665e-03`: #557 measured the *assembled*
gradient as invariant across that whole sweep while the finite-difference reference degraded 28x
between `1e-4` and `4.665e-3`. A trust-region line search consumes the objective surface, not the
gradient. So `auto` is the setting that tells a job what its model's own scale asks for; a job whose
line search turns out to need something tighter still states a number, and this ADR does not claim
otherwise.

### The default path is untouched

- `_derive_atol` and `_derive_atol_vector` take a `loosen` keyword defaulting to `False`, and with
  it false every expression is character-for-character what it was.
- `test_the_derivation_loosens_only_when_asked` covers both sides of the ceiling: three scales at
  or below it return the identical number either way, so every model ADR-0103 moved and every model
  it deliberately left alone is unmoved by the existence of the flag.
- The 23-slug derivation table above was produced by loading each slug twice in one process, unset
  and `auto`, and reading the tolerance each would run at.

### The mechanism, in fixtures

`tests/test_sbml_solver_tolerances.py` gains `_LARGE_SPREAD_SBML` — the ADR-0105 spread lifted six
decades, so `X` starts at 1e+07, `Mid` at 1e+04 and `Tiny` at 1e-03. It is `Weber_BMC2015`'s shape
reduced to something a unit test can hold, and it exhibits the whole defect:

- `test_a_model_above_the_backend_default_gets_no_vector_until_it_opts_in` — the ten-decade model
  takes the *scalar* path under the clamped derivation, which is the "declines exactly where it is
  needed" claim as an assertion. Opting in re-opens the interval and `X` resolves three decades
  apart from `Tiny`.
- `test_the_looser_tolerance_is_what_lets_the_large_model_integrate` — integrator steps, the
  fixture-scale version of the cost claim.
- `test_the_opted_in_model_still_integrates_to_its_closed_form` — the accuracy half, against the
  exact solution: `X` decaying through three time-gated stages from 1e+07 matches `exp` to a
  relative 1.0e-06, which is `rtol` doing what `rtol` is for. That is the shape of the trade — the
  absolute term stops governing and the relative one governs, which is the condition ADR-0103 says
  the derivation is trying to reach.
- `test_tracking_resolves_the_species_a_vector_of_initial_values_cannot` — the tracking oracle, on
  the ADR-0105 fixture where `X` starts at 10 and ends near 7e-17. Its tail under the derived
  vector is pure noise (the vector gives it `1e-8`, eight decades above where it has got to);
  tracking follows it down. This is a *different* claim from the loosening, and is asserted
  separately.
- `test_tracking_at_depth_zero_is_the_loosened_vector` — the strict-extension property, asserted as
  a trajectory rather than as a ceiling, at the ~1 ulp bngsim documents for the three routes.

### Scope

**In:** `_bngsim_caps.py` (`BNGSIM_HAS_TRACKING_ATOL`); `bngsim_sbml_model.py`
(`parse_atol_setting`, the `loosen` keyword on both derivations, `_atol_loosens`, the tracking
branch of `_run_tolerance_kwargs`, and the constructor's shape/capability checks); `parse.py` (the
mode grammar, tried ahead of `numgram`, which error-stops after its `=` and so cannot backtrack);
`config_schema.py`; `config.py`; `docs/config_keys.rst`.

**Out (unchanged):** every BNGL/net model. Every stochastic run. The RoadRunner backend. Every job
that does not state `sbml_atol`, and every job that states a number.

Relevant ADRs: **0103** (the scalar and the clamp this makes liftable), **0105** (the vector, its
floor, and the steady-state pairing), **0086** / **0052** / **0104** (the steady-state and
pre-equilibration paths the explicit `steady_state_tol` protects). Relevant issues: lanl/bngsim
**#196** (the vector), **#212** (its public export), **#213** (the trajectory-following successor),
**#305** (the crossing-stop fix that changed Weber's measurement). Closes issue **#557**; opens
**#586**.
