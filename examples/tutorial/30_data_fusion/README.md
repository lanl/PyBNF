# Lesson 30 — Fusing heterogeneous data in one fit

**Feature:** one fit to multiple experiments of *different data types* — time-course + steady-state + qualitative · **Difficulty:** ★★★ · **Tier:** recovery

Real parameter estimation rarely rests on a single dataset. You have a time course
from one assay, a steady-state dose-response from another, and facts you
qualitatively know to be true. PyBNF fits **one** parameter set to **all** of them
at once — you list several `experiment:` lines, each with its own data file and its
own protocol, and every experiment shares the fitted parameters. This lesson fuses
three genuinely different kinds of data on one reversible reaction.

## The model, and why one dataset isn't enough

A [reversible conversion](reversible_conversion.bngl) `A ⇌ B` (forward rate `kf`,
reverse `kr`). It relaxes to an equilibrium set by the *ratio* of the rates:

```
dA/dt = -kf·A + kr·B         B_eq = kf/(kf+kr) · (A0+B0)      relaxation rate = kf + kr
```

The two rates enter in two separable ways, and that is the whole point:

- an **equilibrium** measurement sees only `kf/kr` (the equilibrium constant) — it
  *cannot* tell a fast pair `(kf, kr)` from a slow pair with the same ratio;
- a **kinetic** measurement sees the relaxation rate `kf + kr` — the timescale the
  equilibrium is blind to.

Neither pins both rates alone. Fused, the ratio and the rate together give `kf` and `kr`.

## Three data types, one conf

```
experiment: relax,       data: relaxation.exp          # a TIME COURSE  (independent var: time)
experiment: titration,   data: titration.exp           # STEADY STATE   (independent var: A0)
experiment: qualitative, data: qualitative.prop, t_end: 8   # QUALITATIVE facts (BPSL)
uniform_var = kf  0.05  3.0
uniform_var = kr  0.02  2.0
```

PyBNF infers each experiment's protocol from its data file — no hand-written protocols:

| experiment | file | independent variable | what PyBNF does |
|---|---|---|---|
| `relax` | [relaxation.exp](relaxation.exp) | `time` | a time course; `Obs_B` rises to equilibrium |
| `titration` | [titration.exp](titration.exp) | `A0` (a parameter) | a **steady-state scan** — runs each total to steady state, reads equilibrium `Obs_B` |
| `qualitative` | [qualitative.prop](qualitative.prop) | — | BPSL constraints, scored by trajectory violation ([lesson 1](../01_logistic_growth)) |

The three contributions add into one objective, minimized over the shared `(kf, kr)`.

## What each experiment buys you

The titration is a clean **structural** demonstration. Fit it *alone* and you recover
the equilibrium constant perfectly — `kf/kr` lands right on `0.7/0.2` — but the
individual rates wander (any fast/slow pair with that ratio fits equally well). It is
non-identifiable on its own, exactly the way [lesson 2](../02_bateman_chain)'s A-only
fit can't see `k2`.

Add the **relaxation** time course and the degeneracy breaks: the kinetics fix
`kf + kr`, the titration fixes `kf/kr`, and together they pin `kf ≈ 0.7`,
`kr ≈ 0.2`. The **qualitative** facts (`B` starts near zero, `B` overtakes `A`, `B`
plateaus high) are a soft guardrail that keeps the fit in the physically sensible
region. Running the committed conf recovers both rates on the nose — which is what
[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py) checks.

Try it yourself: comment out the `relax` line and re-run — the fit keeps the right
ratio but the individual rates drift off.

## Run it

```bash
pybnf -c data_fusion.conf
```

## Where this sits

- [Lesson 9](../09_experiment_design) — the steady-state dose-response and
  pre-equilibration protocols, one at a time; this lesson combines protocols.
- [Lesson 16](../16_joint_fit) — a joint fit across multiple *models*; this is a
  joint fit across multiple *data types* on one model.
- [Lesson 1](../01_logistic_growth) — qualitative (BPSL) constraints on their own.

## Regenerating the data

```bash
python ../regenerate_data.py 30_data_fusion
```

`relaxation.exp` is the model's `Obs_B` at the true rates over time; `titration.exp`
is the equilibrium `Obs_B` at each total `A0`, generated through the real
steady-state scan protocol. `qualitative.prop` is committed by hand (facts, not
model output). Truth lives test-side in [`_manifest.py`](../_manifest.py).
