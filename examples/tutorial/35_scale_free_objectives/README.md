# Lesson 35 — Scale-free objectives (relative vs absolute error)

**Feature:** the normalized / scale-free objective tokens `norm_sos`, `ave_norm_sos`, `sod` (vs plain `sos`) · **Difficulty:** ★★★ · **Tier:** recovery

Every fit so far minimized `sos` — the sum of *squared absolute residuals*. That is
the right objective when the measurement noise is the same size everywhere. But
when the data spans **orders of magnitude**, absolute error quietly throws away the
most informative points, and the fit is dragged off. This lesson shows the problem
and the objectives built to fix it.

## The problem: one decay, three orders of magnitude

[`decay.bngl`](decay.bngl) is a single first-order decay `A(t) = A₀·e^{−k t}` from
`A₀ = 1000`, followed until `A ≈ 1` (three orders of magnitude). The noise is
**multiplicative** — a constant 15% of the value — so in *absolute* terms the early
large points scatter by ±150 while the late small points scatter by ±0.2.

Now think about where the rate `k` is best determined: the whole log-linear decay,
early **and** late. But `sos` sees only the *absolute* residuals, which are ~1000×
bigger early than late — so it fits the noisy early points and effectively ignores
the informative tail. The recovered `k` comes out **dragged**.

## Four objectives, one fit

Each conf changes just `objective =` and refits `k` from the same `decay.exp`:

| conf | `objective =` | how it weights a residual | recovers `k = 0.4`? |
|------|---------------|---------------------------|---------------------|
| [`sos`](sos.conf) | `sos` | absolute — `(sim − exp)²` (σ = 1) | **no** — dragged >10% off (large points dominate) |
| [`ave_norm_sos`](ave_norm_sos.conf) | `ave_norm_sos` | ÷ the column's **mean** | **no** — one column ⇒ a constant ⇒ same as `sos` |
| [`sod`](sod.conf) | `sod` | absolute **L1** — `\|sim − exp\|` | **yes** (~4%) — linear penalty, less dominated by big points |
| [`norm_sos`](norm_sos.conf) | `norm_sos` | ÷ **each point** — `((sim − exp)/exp)²` | **yes** (~1%) — every point weighted equally |

The winner is **`norm_sos`**: dividing each residual by its own measured value
(relative error) makes a 15% miss count the same whether the value is 1000 or 1, so
the informative tail is fully used. `sod` (an L1 penalty) helps too — a linear
penalty is far less dominated by the big early residuals than L2 — but it is still
*absolute*, so it recovers less cleanly than the genuinely relative `norm_sos`.

## `norm_sos` vs `ave_norm_sos` — two different "normalized"

Both are "normalized least squares," but they normalize by different things, and the
distinction matters:

- **`norm_sos`** divides by **each point's own value** — per-point relative error.
  This is the tool for a *single* signal that spans orders of magnitude (this
  lesson).
- **`ave_norm_sos`** divides by the **column's mean** — one scalar per observable.
  That does nothing within a single column (a constant factor cancels, so it
  reduces to `sos`, hence its drag here), but across **several observables of very
  different magnitude** it makes each contribute comparably — the tool for a
  multi-reporter fit where a bright channel would otherwise dominate a dim one.

## Other objective tokens

The [objective surface](../08_robust_objectives) has a few more scale/noise
specifiers worth knowing:

- **`chi_sq_dynamic`** — a Gaussian whose σ is *estimated* rather than read from the
  data: `objective = chi_sq_dynamic` plus a declared `uniform_var = sigma__FREE <lo>
  <hi>`. Reach for it to fit the noise level jointly with the parameters when you
  have no `_SD` column and the noise is (roughly) constant.
- **`lognormal`** — a Gaussian in `log10` space (multiplicative noise as an
  additive log model), using each point's `_SD`.
- **`sod`** is the token form of an L1 fit; the [robust-objectives lesson](../08_robust_objectives)
  reaches the same Laplace/L1 idea through the `noise_model` surface with an
  *estimated* scale, and adds Student-t.

## Where this sits

- [Lesson 8](../08_robust_objectives) — robust objectives (`noise_model` =
  Gaussian/Laplace/Student-t) for **outliers**; this lesson is about **scale**.
- [Lesson 10](../10_per_observable_noise) — give each observable its own noise
  model (another way to stop one channel dominating).
- [Lesson 22](../22_normalization) — normalize the *simulation* to match data
  reported relative to a reference (a related but distinct "scale-free" idea).

## Regenerating the data

```bash
python ../regenerate_data.py 35_scale_free_objectives
```

The `.exp` is the decay at the true `k`, corrupted with seeded 15% multiplicative
noise (`noise_cv` in [`_manifest.py`](../_manifest.py)) — huge absolute scatter on
the large points, tiny on the small, which is exactly what separates the objectives.
