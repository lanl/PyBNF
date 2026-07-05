# Lesson 46 — Model checking: does the model satisfy the spec?

**Feature:** `job_type = check` + BPSL `.prop` specifications · **Difficulty:** ★★ (recovery tier)

Every other lesson *fits* — it searches parameter space for values that match data.
This one does the opposite. **Model checking** takes a model exactly as written —
its built-in parameter values, no free parameters — and asks a yes/no question of
each of several qualitative **properties**: does the trajectory do what it is
supposed to? It runs **one** simulation and no search. This is how you turn a
model into a testable object: a `.prop` file is a spec, and `check` is the
pass/fail gate that runs it.

Lesson 1 showed a one-line taste of this (a model that passes all its properties).
Here we make it *discriminating*: one specification, two models, and `check` tells
them apart.

## The model: a signaling pulse that should rise, peak, and clear

A stimulus turns an inactive precursor `Pre` into an active response `R`, which is
then cleared away (`Pre → R → Clr`, observing `R`). The active species rises then
falls — a **pulse**. A *healthy* response has three qualitative features:

* it **starts** near baseline,
* it **mounts a real response** (peaks well above baseline) but without
  **overshooting** a safe ceiling, and
* it **clears** back to baseline afterwards.

[`signaling_pulse.bngl`](signaling_pulse.bngl) is the healthy circuit (fast
activation `kact = 1.5`, brisk clearance `kclr = 0.5`): the pulse peaks near 58 and
is gone by `t = 8`. [`signaling_pulse_impaired.bngl`](signaling_pulse_impaired.bngl)
is identical **except** clearance is knocked down (`kclr = 0.03`) — the lesion. Now
the response barely drains: it overshoots to ~92 and is still ~75 at the end.

## The spec: five properties, the whole where-vocabulary

[`pulse.prop`](pulse.prop) is written in PyBNF's **Biological Property Specification
Language (BPSL)**. Each line is `<expr> <op> <expr> <where>`, and the five lines
deliberately exercise every `where` clause:

| # | Property | `where` | Reads as |
| --- | --- | --- | --- |
| 1 | `Obs_R < 5`  | `at time=0` | starts at baseline |
| 2 | `Obs_R > 50` | `once` | true at least once → mounts a real response |
| 3 | `Obs_R < 75` | `always` | true at every point → never overshoots the ceiling |
| 4 | `Obs_R > 20` | `at time=3` | still elevated partway through (a sustained pulse) |
| 5 | `Obs_R < 10` | `between time=8,time=12` | true across the tail → clears by the end |

* **`at time=v`** checks one time point. **`once`** passes if the inequality holds
  *anywhere* on the run (a peak). **`always`** requires it at *every* output point (a
  ceiling). **`between a,b`** requires it across an interval (a settled band).

## Run the check on both models

```bash
pybnf -c signaling_pulse_check.conf            # the healthy circuit
pybnf -c signaling_pulse_impaired_check.conf   # the impaired one
```

The healthy circuit passes everything:

```
Objective value is 0.0
Satisfied 5 out of 5 constraints
```

The impaired circuit fails the two properties clearance was protecting — the
ceiling (property 3) and the return to baseline (property 5):

```
Objective value is 87.6
Satisfied 3 out of 5 constraints
```

Same spec, two models, a clean pass/fail verdict — that is `check` as a QA gate.

## Which property failed? Read the itemized penalties

A `check` writes one file per `.prop`, **one line per property, in file order**:
`0` for satisfied, a positive penalty for violated. (Note it lands in
`Simulations/`, not `Results/`.) For the impaired model,
`output/signaling_pulse_impaired_check/Simulations/qualitative_constraint_eval.txt`
is:

```
0.0                    <- 1  Obs_R < 5  at time=0        satisfied
0.0                    <- 2  Obs_R > 50 once             satisfied
17.32                  <- 3  Obs_R < 75 always           VIOLATED (overshoots)
0.0                    <- 4  Obs_R > 20 at time=3        satisfied
70.27                  <- 5  Obs_R < 10 between 8,12      VIOLATED (never clears)
```

The nonzero lines point straight at the two failures, and their magnitudes measure
*how badly* (the peak overshoots the ceiling by ~17; the tail sits ~70 above where
it should have cleared to).

## Notes / gotchas

- **`check` has no free parameters.** It evaluates the model at its written-in
  values, so to check a *different* parameterization you supply a different model
  (here, the impaired sibling) — not a fit conf. That is why this lesson ships two
  `.bngl` files that differ in one rate.
- **One spec, either model.** A bare observable (`Obs_R`) in a `.prop` resolves to
  whichever model the experiment runs, so the *same* `pulse.prop` serves both checks.
- **It's a deterministic, single-trajectory tool.** `check` evaluates one ODE
  trajectory and reports pass/fail per property. It does *not* run stochastic
  replicates and report a satisfaction *fraction* — so use it on ODE models. (For
  weighted or probabilistic properties, BPSL also supports `weight <w>` and a
  likelihood form with `confidence`/`pmin`/`pmax`; see `pybnf/constraint.py`.)
- **Checking vs. fitting qualitative data.** The same BPSL properties can instead be
  *fit* — searching for parameters that satisfy them — which is what lesson 1's
  `logistic_growth_constraints.conf` and lesson 30 (data fusion) do. `check` is the
  no-search flip side: score the model you already have.

## The test

[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py)
(`test_tutorial_model_check_discriminates`, recovery tier) runs `check` on both
confs, asserts the exact `Satisfied 5 out of 5` / `Satisfied 3 out of 5` lines, and
reads `qualitative_constraint_eval.txt` to confirm the penalties pin the violated
properties (lines 3 and 5) for the impaired model.
