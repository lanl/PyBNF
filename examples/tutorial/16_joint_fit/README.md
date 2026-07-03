# Lesson 16 — A multi-experiment joint fit (shared parameters)

**Feature:** fitting several experiments at once with shared free parameters (multi-model, ADR-0034/0041) · **Difficulty:** ★★★

Real data rarely comes from a single experiment. You dose a drug two ways, or run
a control and a treatment, or measure at two temperatures — and the *same*
underlying parameters explain all of it. The right thing to do is fit every
experiment **together**, against one shared parameter set, so the estimate is
consistent with all the evidence at once. That is a **joint fit**.

Here the system is a **two-compartment pharmacokinetic** model — a drug moving
between plasma (central) and tissue (peripheral) and being eliminated:

```
dCentral/dt    = -(k12 + ke)·Central + k21·Peripheral
dPeripheral/dt =        k12 ·Central - k21·Peripheral
```

The three rates `k12`, `k21`, `ke` are properties of the drug. We observe the
**plasma** concentration in two experiments that differ only in where the dose
starts:

| Experiment | Model | Dose | Plasma curve |
| --- | --- | --- | --- |
| `central` | [`central_bolus.bngl`](central_bolus.bngl) | all in plasma (IV bolus) | starts high, **biexponential decay** |
| `peripheral` | [`peripheral_bolus.bngl`](peripheral_bolus.bngl) | all in tissue | starts at zero, **rises then falls** |

Those two curves look nothing alike ([`central_dose.exp`](central_dose.exp) vs
[`peripheral_dose.exp`](peripheral_dose.exp)) — yet one `{k12, k21, ke}` explains
both. The central route mostly constrains `k12 + ke`; the peripheral route's
*rise* is a sharp handle on `k21`. Fitting them jointly determines all three more
tightly than either route alone.

## Sharing a parameter across models

The two experiments use two different model files (they differ only in the seed
species — which compartment the dose lands in). How does the fit know they share
the same rates? Because **new-era PyBNF binds a free parameter to a model
parameter by its bare id** (ADR-0034): declaring

```
uniform_var = k12  0.05  3.0
```

binds `k12` to the parameter named `k12` in *every* model that has one. So the one
declaration ties `central_bolus.bngl`'s `k12` and `peripheral_bolus.bngl`'s `k12`
into a single fitted quantity. Each experiment names the model it runs against:

```
model: central_bolus.bngl
model: peripheral_bolus.bngl

experiment: central,    model: central_bolus.bngl,    data: central_dose.exp
experiment: peripheral, model: peripheral_bolus.bngl, data: peripheral_dose.exp
```

PyBNF simulates each experiment against its own model, scores it against its own
data, and **sums** the two objectives — one number the optimizer minimizes over
the shared `{k12, k21, ke}`.

## Run it

```bash
pybnf -c joint_fit.conf
```

Differential evolution (with a `refine` polish) recovers all three rates —
`k12 = 0.8`, `k21 = 0.4`, `ke = 0.3` — from the two curves together.

## Why not just fit each experiment separately?

You could, but then you'd get two `ke` estimates that disagree by a little (noise
pulls them in different directions), and no principled way to combine them. A joint
fit gives **one** parameter set that is the best explanation of *all* your data,
and it borrows strength across experiments: a parameter poorly constrained by one
curve can be well constrained by another. That is the whole reason to run more than
one experiment.

## Regenerating the data

Each `.exp` is its model's own plasma output at the true rates:

```bash
python ../regenerate_data.py 16_joint_fit
```

The true rates and the recovery tolerance live test-side in
[`_manifest.py`](../_manifest.py); the joint fit is checked by
[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py)
(`recover` mode).
