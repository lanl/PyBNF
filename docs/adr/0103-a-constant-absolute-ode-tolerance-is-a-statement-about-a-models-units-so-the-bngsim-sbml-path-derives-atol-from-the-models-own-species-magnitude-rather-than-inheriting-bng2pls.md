# A constant absolute ODE tolerance is a statement about a model's units, so the bngsim SBML path derives `atol` from the model's own species magnitude rather than inheriting BNG2.pl's (issue #546)

**Status: Accepted and implemented (2026-08-06).** `Giordano_Nature2020`'s gradient was wrong on
41 of its 50 fitted parameters, by up to 26%, with no refusal and no warning. The cause is not the
model's piecewise-in-time structure, which the backend already handles; it is that bngsim's
absolute solver tolerance — BNG2.pl's `1e-8` — is larger than that model's entire state.

## The problem, and what it was not

Issue #546 opened on a plausible reading. `Giordano_Nature2020` is a COVID NPI model whose 14
assignment rules carry 110 `piecewise` expressions, every one gated on simulation time
(`alpha = piecewise(alpha_0, time <= 4 + initialTimeManual) + …`). Its assembled gradient
disagreed with central differences at every step size tried — the corpus's own test for a real
defect (`tools/README.md`: a real defect does not move with `h`, FD noise does) — and the
disagreement partitioned along whether a parameter sat behind a time gate:

| group | n | median rel err | max | columns > 1e-2 |
|---|---:|---:|---:|---:|
| not time-gated | 9 | 9.2e-06 | 9.3e-03 | 0 |
| time-gated | 41 | 1.3e-03 | 2.6e-01 | 10 |

The reading that follows is that the integrator is stepping over discontinuities nobody told it
about. It is not. **bngsim's SBML loader already collects every inequality against the `time`
csymbol and registers it as a CVODE root** (`_collect_time_discontinuity_conditions` ->
`add_discontinuity_trigger`). Giordano gets 13 of them, one per distinct NPI stage edge; the
fixture added with this ADR gets 4. The boundaries are landed on exactly, on the state solve and
the sensitivity solve alike, and the model has no state-jumping event at all — so none of the
#461/#536 discrete-event line applies to it either.

What is actually wrong is the **absolute** tolerance. CVODE's error test weights each state by
`rtol*|y_i| + atol`, so a constant `atol` is a declaration that values beneath it are noise — a
statement about the *model's units*, not a universal constant. Giordano is a population-*fraction*
epidemic model: its species sit between `1.7e-8` and `1`, with a median of `3.7e-7`. At
`atol = 1e-8` the absolute term buries the relative one across the whole early trajectory, which
therefore carries no significant digits. The forward-sensitivity solve carries fewer still, since
CVODES scales the state tolerances by the parameter magnitude for the sensitivity vectors — for a
fitted rate of ~0.1 their absolute floor is ten times *looser* than the states'.

That also explains the partition, without the gate being the cause. A gated parameter's whole
influence is confined to one stage window, and the earliest, narrowest windows are exactly where
the states are smallest; an ungated parameter either never enters the ODE (the seven `sd_*` noise
scales) or accumulates over the full 45 days, where the states have grown by five decades and the
relative term governs. The correlation #546 measured is real. Its cause is where each parameter
acts, not the switching.

The decisive evidence is that the disagreement moves with `atol` **alone**. Same point, same step
size, worst relative error over all 50 columns:

| | `atol=1e-8` | `atol=1e-12` | `atol=1e-14` |
|---|---:|---:|---:|
| `rtol=1e-8` | 7.7e-02 | 1.2e-02 | 5.2e-04 |
| `rtol=1e-12` | 7.2e-02 | — | 1.9e-04 |

Four decades of `rtol` buy nothing.

The reason a PEtab-imported job could not simply ask for something else is that it had no way to:
`net_model.py` reads `atol`/`rtol` out of a BNGL actions block, and `bngsim_sbml_model.py` never
passed either to `Simulator.run`. The documented position (`docs/config_keys.rst`) was that a BNGL
author writes `atol=>…` in the actions block — which is right, and leaves an SBML author, whose
model has no actions block, with nothing.

## The decision

**Derive `atol` from the model, on the path that has no other way to state it, and let it only
ever tighten.**

- **The scale is the median strictly-positive nominal species value**
  (`_compute_nominal_state_scale`), read off the parsed SBML document, in the concentration units
  bngsim integrates in (`_species_unit_factor`, the same conversion the in-place value paths
  apply).

- **The tolerance is `clamp(rtol * scale, 1e-16, backend_default)`** (`_derive_atol`). The upper
  clamp is what makes this safe to apply to every SBML model rather than only the ones that need
  it: nothing is ever loosened, so a model of order-one scale keeps `1e-8` bit-for-bit and every
  existing trajectory is unchanged. Across the 23-slug subset-I corpus, 19 models are untouched
  and 4 tighten — Giordano by seven decades, Brannmark by 1.5, Bertozzi and Armistead by less than
  one.

- **Median, not minimum.** The first implementation used the minimum, and `Brannmark_JBC2010`
  withdrew it within the hour. That model seeds one transient intermediate (`IRp`) at `1.8e-9`
  while its principal species sit at `0.1..10`, so the minimum rule asked for `atol = 1e-17`; at
  interior fit points CVODE then exhausted `mxstep` and the **simulation failed outright**, in
  seconds rather than milliseconds, on a model that had been integrating fine. Trading a wrong
  gradient for a dead one is not a fix. The median asks the question the tolerance is actually
  about — what magnitude is this model written in? — and is unmoved by a single outlier at either
  end: Brannmark reads `0.033` and stays near the default, Giordano reads `3.7e-7` and does not.

- **The derivation is a constant of the model, not of the fit point.** It is computed from the
  document at load and memoized, never from the state a given evaluation is about to integrate.
  A tolerance that moved with a fitted initial condition would put a step in the objective
  wherever the derivation crossed a rounding boundary; worse, the scalar (line-search) evaluations
  of a gradient fit must be integrated to the same accuracy as the sensitivities they are compared
  against, or the two disagree about what the objective *is*.

- **`sbml_rtol` / `sbml_atol` state either explicitly**, for the model that needs something the
  derivation will not reach for. Both are bngsim-only — RoadRunner has its own integrator
  settings and the BNGL path has the actions block — and setting either under another backend is
  a config error rather than a silent no-op. When the `1e-16` floor binds, that is logged once per
  model with `sbml_atol` named, because the one case the derivation cannot fully serve must not be
  silent in the way the original bug was.

## Scope

**In:** `bngsim_sbml_model.py` (`_derive_atol`, `_compute_nominal_state_scale`,
`_effective_tolerances`, and the `rtol`/`atol` kwargs on `_run_simulation`'s `sim.run` call);
`bngsim_antimony_model.py` (the constructor pass-through); `config_schema.py` / `parse.py` /
`config.py` (the two keys, their validation, and their route into both model classes);
`docs/config_keys.rst`.

**Out (unchanged):** every BNGL/net model — its tolerances come from BNG2.pl's actions block and
its parity with BNG2.pl is what that backend is measured against. Every stochastic (`ssa`) run,
which has no CVODE tolerances to set, so its `run` call is byte-identical. Every SBML model whose
median species value is at or above one, which is most of them. The RoadRunner SBML backend.

**Carried along deliberately:** a **steady-state** run's convergence cutoff. bngsim falls its
`steady_state_tol` back to `atol` when unset (as BNG2.pl does), so a derived tolerance tightens
the `||dx/dt||` test too. That is the coherent direction — the same statement about the model's
magnitude governs both, and the alternative is a relaxation criterion looser than the integration
that produced it. It also fixes the degenerate case it used to produce: for a model whose states
are ~1e-8, `||dx/dt|| < 1e-8` is satisfied *at t = 0*, so ADR-0086's "relax to equilibrium"
returned the initial state and called it the steady state. The cost is that such a relaxation now
runs until it is actually flat, bounded as before by the run's max-time bound and `wall_time_sim`,
with ADR-0046's warn-and-score-the-last-value if it does not get there.

**Deliberately out:** a *per-species* absolute tolerance, which is what CVODE's `SVtolerances`
exists for and would be strictly better than any scalar — each state resolved against its own
magnitude, with no cross-species over-tightening and no need for a robust statistic at all.
bngsim's `Simulator.run` takes a scalar `atol`, so this needs a backend change first. Deriving the
scale from the *measurement* magnitudes rather than the model's initials, which is arguably the
better signal for a fit (it is what the residuals are computed against) but couples the model
object to the experimental data it does not currently see. Neither is needed for #546.

## Verification

- **A piecewise-in-time analytic oracle** (`tests/test_sbml_solver_tolerances.py`) — the fixture
  of this shape the whole #385 gradient epic never had, which is why it stayed green through this.
  `X' = -k(t)·X` with `k` switching across three stages, each stage its own single-piece
  `piecewise` summed with the others (Giordano's idiom) and the middle condition an `and` of two
  inequalities, so bngsim declines the analytic sensitivity RHS and falls back to CVODES' internal
  difference quotient — Giordano's exact code path. `X(0) = 1e-8` puts the trajectory at the old
  default. The closed form gives an *exact* sensitivity oracle, `∂X/∂k_j = -w_j(t)·X(t)` with
  `w_j` the time spent in stage `j`, which is a stronger arbiter than finite differences.
  Asserted positively (every column within 1e-4), and with the negative control: the same model
  pinned back to `sbml_atol = 1e-8` is wrong by more than 10% on every column.
- **The boundaries are already roots**, asserted rather than assumed: the fixture registers four
  discontinuity triggers and reports no discrete events.
- **`Brannmark_JBC2010`'s shape is a test**, not a scar — one negligible transient nine decades
  under its principal species must not set the tolerance for the model around it.
- **The corpus**, at an interior bounds-clear point, `h = 1e-3`, worst relative error over every
  fitted column: Giordano 7.70e-02 -> 4.45e-04; Brannmark 4.98e-05 -> 3.56e-05; Bertozzi 2.74e-05
  -> 2.35e-05; Armistead 4.20e-06 -> 4.19e-06; the untouched controls (Boehm, Weber) unmoved.
  Giordano's evaluation is ~14% slower; no other slug's runtime moved measurably, and none failed.
- The full default suite is green, with the same failure set as `main` on this machine (the
  BNG2.pl-dependent tests, which need a BNG2.pl this host does not have).

Relevant ADRs: **0093** (the wall-clock budget, which a tolerance change spends against),
**0086** (steady-state runs, which take the same derived pair). Relevant issues: **#536** and
**#461** (the discrete-event sensitivity line, which this model is *not* on), **#535** (the FD
sweep that found this), **#385** (the gradient epic whose fixture set this fills a hole in).
Closes issue **#546**.
