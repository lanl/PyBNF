# Lesson 19 — Fitting a shape (column-joint profile objectives)

**Feature:** `profile_objective = kl` / `wasserstein` (scale-free shape matching) · **Difficulty:** ★★★

Every objective so far compared the model to the data **point by point**. But
sometimes the absolute scale of a signal is meaningless — a detector reads out in
arbitrary fluorescence units, gain drifts between runs, a readout is normalized —
and all the information lives in the **shape** of the curve. Point-by-point scoring
can't cope: an unknown multiplier on the data wrecks every residual. This lesson
fits the shape instead, with a **column-joint** objective.

## The scenario

A transient pulse — `A --k1--> B --k2--> C`, observing `B(t)` (`pulse.bngl`). The
middle species rises then falls; the two rates set the pulse's **shape** (rise
speed, peak position, tail), while the initial amount and any detector gain set only
its **height**. The committed data [`pulse_shape.exp`](pulse_shape.exp) is that pulse
in **arbitrary units** (scaled up ×1000) — the height is deliberately meaningless.

## Two shape objectives, one line each

A profile objective normalizes both the simulated and measured pulse to a
probability distribution over time and scores how far apart those **distributions**
are — so any overall multiplier on the data cancels out.

| Conf | Objective | What it measures |
| --- | --- | --- |
| [`pulse_kl.conf`](pulse_kl.conf) | `profile_objective = kl` | Kullback–Leibler cross-entropy (a statistical, likelihood-flavored shape match) |
| [`pulse_wasserstein.conf`](pulse_wasserstein.conf) | `profile_objective = wasserstein` | 1-Wasserstein / earth-mover distance (a geometric shape match) |

```bash
pybnf -c pulse_kl.conf
pybnf -c pulse_wasserstein.conf
```

Both recover `k1` and `k2` from the pulse shape alone, despite the arbitrary units —
multiply `pulse_shape.exp` by any constant and the fit is unchanged. The two objectives
are the two ends of the `profile_objective` family: `kl` is a **likelihood**, `wasserstein`
is a **distance**, and they land on the same answer from opposite definitions.

## Things worth knowing

- **Scale-free, whole-fit.** A profile objective is its own objective, so it takes
  **no** per-point `noise_model` and **no** `_SD` column. It is mutually exclusive
  with `objective = …`.
- **KL vs Wasserstein conditioning.** `wasserstein` normalizes *both* profiles, so
  its score is O(1) and it converges tightly here. `kl` weights by the (un-normalized)
  measured column, so its score inherits the data's magnitude and converges a touch
  looser — visible as a slightly wider recovery tolerance in the tests. Normalizing or
  rescaling the data tightens it.
- **The flip-flop symmetry.** The pulse shape is symmetric under swapping `k1` and
  `k2` (the classic pharmacokinetic "flip-flop" ambiguity): from the shape alone you
  cannot tell the rise rate from the fall rate. The bounds in the confs encode the
  prior knowledge that the rise (`k1`) is faster, which selects the intended solution.

## When to reach for this

Whenever the **shape** of a distribution or profile is the data and the amplitude is
not: matching a measured residence-time or transit-time distribution, a normalized
spatial profile, a histogram, or any read-out where only relative values are
trustworthy. Recovery is checked in
[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py).
