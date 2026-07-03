# Lesson 8 — Robust objectives (when a few bad points wreck the fit)

**Feature:** the `noise_model` surface — Gaussian vs Laplace vs Student-t (outlier robustness) · **Difficulty:** ★★☆

Real data has outliers: a mispipetted well, a detector glitch, a mislabeled
sample. The **noise model** you choose — your assumption about what the residuals
look like — decides whether those few bad points quietly bias your answer or get
shrugged off. This lesson fits the *same* contaminated decay two ways and lets you
watch the difference.

## The setup

The data ([`contaminated_decay.exp`](contaminated_decay.exp)) is a clean
exponential decay `A(t) = 100·exp(-0.5·t)` with **three gross outliers** spliced
in (at t = 2, 6, 8 — you can see them tower over the curve). Everything else is
noise-free, so the *only* thing that can pull a fit off the truth is those three
points.

## Two noise models, two outcomes

| Conf | Noise model | Result |
| --- | --- | --- |
| [`decay_gaussian.conf`](decay_gaussian.conf) | `normal, sigma = read_exp_file _SD` | `k` **≈ 0.36** — ~28% off (dragged) |
| [`decay_laplace.conf`](decay_laplace.conf) | `laplace, scale = fit noise_scale` | `k` **≈ 0.50** — on the truth (robust) |
| [`decay_student_t.conf`](decay_student_t.conf) | `student_t, sigma = read_exp_file _SD, df = fix_at 4` | `k` **≈ 0.50** — on the truth (robust, tunable) |

```bash
pybnf -c decay_gaussian.conf     # the default assumption — and the mistake
pybnf -c decay_laplace.conf      # the robust fix
pybnf -c decay_student_t.conf    # robust too, with a tail-heaviness dial
```

The assertions live in
[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py): the
Laplace and Student-t fits must recover `k` to within 3%, and the Gaussian fit must
be *provably* wrong — off by at least 10% — so the comparison is never vacuous.

## Why Gaussian breaks and Laplace doesn't

- A **Gaussian** noise model is a least-squares fit, and squared error is
  dominated by the *largest* residuals. To shrink the enormous squared errors at
  the three outliers, the optimizer bends the whole curve toward them — biasing
  the rate. The Gaussian's thin tails make a 10σ point "impossible," so the fit
  refuses to leave it unexplained.
- A **Laplace** noise model has **heavy tails**: a far-out residual is
  *improbable, not impossible*, so the fit tolerates it. Maximizing a Laplace
  likelihood is least-**absolute**-deviation fitting, whose optimum is
  median-like — and the median famously ignores a few extreme values.
- A **Student-t** noise model is heavy-tailed too, but with a **dial** — the
  degrees of freedom `df`. Large `df` *is* a Gaussian (least squares, not robust);
  small `df` gives progressively heavier tails (`df → 1` is a Cauchy). At `df = 4`
  the outliers are tolerated and `k` comes back on the truth, and you can turn the
  robustness up or down explicitly instead of taking whatever Laplace gives you.
  (`df` defaults to 4; you *can* estimate it with `df = fit nu`, but it is only
  weakly identified, so a fixed small value is the usual choice.)

## The new-era `noise_model` surface

Both confs use edition-2's `noise_model` line, which names the noise family and
says where each of its parameters comes from:

```
noise_model = normal,  sigma = read_exp_file _SD      # fixed, from the data column
noise_model = laplace, scale = fit noise_scale        # estimated, a free parameter
```

The `scale = fit noise_scale` clause introduces a **nuisance parameter** — the
Laplace spread, estimated alongside the rate (declared with an ordinary
`loguniform_var` line, no special `__FREE` naming). Other families and sources are
available the same way — e.g. `sigma = fix_at 2`, or `dispersion = fix_at 10` for a
`neg_bin` count model — so the objective is assembled declaratively from the noise
family up.

## The takeaway

Least squares (a Gaussian noise model) is the right default *only when the noise is
actually Gaussian*. The moment your data can contain outliers, reach for a
heavy-tailed noise model — `laplace`, or `student_t` when you want to dial the
robustness yourself — and let the fit decide which points to trust.
