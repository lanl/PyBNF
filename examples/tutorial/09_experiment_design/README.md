# Lesson 9 — Experiment design (conditions, dose-response, pre-equilibration)

**Feature:** the `condition:` / `preequilibrate:` surface; steady-state parameter scans · **Difficulty:** ★★★

So far every lesson fit a single time course. Real studies use richer *experiment
designs*: sweep a dose and read a steady state, or pre-incubate under one condition
and measure the relaxation under another. PyBNF lets you **declare the design** and
synthesizes the simulation protocol for you — you never hand-write a scan or a
multi-phase action. This lesson fits one model
([`inducible_gene.bngl`](inducible_gene.bngl)) two ways, and both recover the same
degradation rate `k_deg`.

The model is an inducible producer: `dA/dt = k_prod·Stimulus_isOn − k_deg·A`, whose
stimulus-on steady state is `A_ss = k_prod/k_deg`.

## Design 1 — dose-response at steady state

[`dose_response.conf`](dose_response.conf) sweeps the stimulus strength `k_prod`
and reads the steady-state level at each dose. The trick is in the data file: its
independent-variable column is **`k_prod`, not `time`**
([`dose_response.exp`](dose_response.exp)), so PyBNF infers a **parameter scan** —
and because no measurement time is given, it takes each dose to **steady state**
(bngsim solves for the fixed point directly, no long transient to integrate).

```
experiment: doseresponse, data: dose_response.exp      # indvar k_prod → steady-state scan
```

`A_ss = k_prod/k_deg` is a straight line through the origin whose slope is
`1/k_deg`, so the five doses pin `k_deg` down cleanly.

## Design 2 — a two-phase washout (pre-equilibration)

[`washout.conf`](washout.conf) pre-incubates with the stimulus **on** until steady
state, then switches it **off** and watches `A` relax as `A_ss·exp(−k_deg·t)`. You
declare two conditions and tell the experiment to equilibrate under one before
measuring under the other:

```
condition: stim_on,  perturbations: Stimulus_isOn = 1
condition: stim_off, perturbations: Stimulus_isOn = 0
experiment: washout, preequilibrate: stim_on, condition: stim_off, data: washout.exp
```

PyBNF synthesizes the two-phase protocol: **equilibrate** (`stim_on`, run to steady
state, unmeasured) → **switch** → **measure** (`stim_off`, over the data grid). The
crucial part is **carry-over**: the measurement begins at the equilibrated level
(`A(0) = 1.5` in [`washout.exp`](washout.exp)), *not* the model's seed `A = 0`. If
state didn't carry across the switch, the measurement would start at zero and stay
flat — matching nothing. Recovering `k_deg` here proves the carry-over works.

## What to notice

- **The protocol is generated, not written.** The model has no `begin actions`
  block. A steady-state scan and a two-phase equilibrate-then-measure both come out
  of the *experiment declaration* — the same model file serves both designs.
- **`time` is special — anything else is a dose.** Naming the data's independent
  column after a model parameter is all it takes to turn a fit into a
  dose-response scan.
- **Steady state is solved, not simulated.** With no `t_end`, bngsim finds the
  fixed point with a Newton/KINSOL solve instead of integrating out a long
  transient — faster and exact.
- **One rate, two experiments.** Both designs recover `k_deg = 2.0` (asserted in
  [`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py)) —
  a reassuring cross-check that the designs, and PyBNF's protocol synthesis, agree.
