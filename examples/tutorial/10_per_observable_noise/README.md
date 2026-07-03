# Lesson 10 — Per-observable noise (each reporter its own error model)

**Feature:** per-observable `noise_model <obs> = …` overrides · **Difficulty:** ★★★

Lesson 8 chose one noise model for a whole fit. But a real experiment often
measures several readouts *with different instruments* — a clean fluorescent
reporter here, a noisy count-based assay there. Forcing one noise model on all of
them is a compromise. PyBNF lets you give **each observable its own noise model**,
so you can be robust exactly where you need to be.

## The setup

The model ([`two_reporter.bngl`](two_reporter.bngl)) is a chain `A → B → C` watched
through two reporters, and the two rate constants lean on different ones:

- **Obs_A** is clean, and `A(t) = A0·exp(−k1·t)` depends on **k1 alone** — so Obs_A
  pins `k1` down tightly.
- **Obs_C** is where **k2** hides (C can only rise as fast as B→C allows) — but this
  reporter is noisy and carries three gross **outliers** (see
  [`reporters.exp`](reporters.exp)).

## Two fits

| Conf | Noise model(s) | Result |
| --- | --- | --- |
| [`both_gaussian.conf`](both_gaussian.conf) | one `normal` for both | `k1 ≈ 0.81` ✓ but **`k2 ≈ 0.19` — ~23% off** |
| [`per_observable.conf`](per_observable.conf) | Obs_A `normal`, **Obs_C `laplace`** | `k1 ≈ 0.80`, `k2 ≈ 0.25` — both recovered |

```bash
pybnf -c both_gaussian.conf      # one model for both reporters — k2 gets dragged
pybnf -c per_observable.conf     # robust only on Obs_C — k2 comes back
```

Notice the split: `k1` is fine in *both* fits — the clean Obs_A pins it no matter
what. It's `k2`, which lives in the contaminated Obs_C, that a single Gaussian
model drags off (least squares chases the outliers). The clean reporter can't
protect a parameter it doesn't constrain — so you have to fix the noisy reporter
directly. (Asserted in
[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py): the
per-observable fit recovers both rates within 3%; the all-Gaussian fit is off by
at least 10%.)

## The per-observable override surface

A whole-fit `noise_model` line sets the default; a line **keyed by observable
name** overrides it for just that observable:

```
noise_model = normal, sigma = read_exp_file _SD     # default — applies to Obs_A
noise_model Obs_C = laplace, scale = fit b_C        # override — applies only to Obs_C
```

Everything from Lesson 8's noise vocabulary is available per observable — a
different family (`normal`/`laplace`/`neg_bin`/…), a different `sigma`/`scale`
source (`read_exp_file`, `fit`, `fix_at`, `relative`, …). So a fit with a
continuous reporter and a count reporter, or a precise channel and a flaky one,
gets each described on its own terms.

## The takeaway

Don't average away what you know about your instruments. When observables differ
in how they're measured, describe each one's noise separately — you keep the clean
channels' precision *and* get robustness on the messy ones, in the same fit.
