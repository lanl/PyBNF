# Lesson 45 — Model selection: which growth law?

**Feature:** comparing competing models by AIC · **Difficulty:** ★★★ (recovery tier)

Every earlier lesson knew the model and fit its parameters. Often you *don't* know
the right model — you have a growth curve and several candidate **laws** it might
follow. The workflow is: fit each candidate to the same data, then compare them
fairly. This lesson fits **four growth laws** to one noisy curve and lets the data
pick the winner.

## The candidates and the data

The data ([`growth.exp`](growth.exp)) is an asymmetric **Richards** curve (shape
exponent `b = 3`, a sharp approach to the carrying capacity) with gaussian noise.
Four candidate models — all observing `Obs_N`, so they fit the same column — compete:

| Model | law | free params (`k`) |
| --- | --- | --- |
| [`logistic.bngl`](logistic.bngl) | `r·N·(1 − N/K)` — symmetric S-curve | `r, K` (2) |
| [`gompertz.bngl`](gompertz.bngl) | `r·N·ln(K/N)` — a fixed skew | `r, K` (2) |
| [`richards.bngl`](richards.bngl) | `r·N·(1 − (N/K)^b)` — a free shape `b` | `r, K, b` (3) |
| [`von_bertalanffy.bngl`](von_bertalanffy.bngl) | `a·N^(2/3) − b·N` — surface-area limited | `a, b` (2) |

## Fit them all the same way

Each conf ([`fit_richards.conf`](fit_richards.conf), …) is identical except for its
`model:` line and parameters, and every one scores the fit with the **same**
weighted least-squares objective:

```
noise_model = normal, sigma = read_exp_file _SD
```

so the best objective — a **chi-square** — is directly comparable across models.

```bash
for m in richards logistic gompertz von_bertalanffy; do pybnf -c fit_$m.conf; done
```

## Rank by AIC (don't just pick the lowest chi-square)

A more flexible model (more parameters) can always match the data at least as well,
so the raw chi-square favours complexity. The **Akaike Information Criterion**
charges for each parameter:

```
AIC = chi_square + 2·k          (k = number of fitted parameters)
```

and you pick the model with the **lowest** AIC. For this data:

| Model | chi-square | `k` | AIC | rank |
| --- | --- | --- | --- | --- |
| **Richards** | **≈ 11** | 3 | **≈ 17** | **1 (true)** |
| logistic | ≈ 62 | 2 | ≈ 66 | 2 |
| gompertz | ≈ 244 | 2 | ≈ 248 | 3 |
| von Bertalanffy | ≈ 363 | 2 | ≈ 367 | 4 |

Richards wins decisively: it fits the asymmetric curve so much better than the
others that its extra shape parameter `b` is *more than* paid for by the `2·k`
penalty. The others are the wrong *shape* and can't be rescued by their smaller
parameter count. (When two models fit almost equally well, AIC would correctly
prefer the simpler one — that is the whole point of the penalty.)

**Selecting the law is not the same as pinning the parameters.** Model selection
picks the right *structure* — here, that the data follow a Richards law. Richards's
three parameters `r`, `K`, `b` partly **trade off** (a sloppy direction), so a fit
can reproduce the curve with a `(r, K, b)` that is not exactly the generating one.
That is a real and common situation — the model can be identified even where its
parameters are only loosely so — and it is a reason to reach for the identifiability
tools of lessons 02 (profile likelihood), 05 (bootstrap) and 17 (posterior) *after*
you have chosen the model.

## The test

[`tests/test_tutorial_model_selection.py`](../../../tests/test_tutorial_model_selection.py)
(recovery tier) fits all four candidates through the faked-dask harness, computes
each AIC, and asserts that Richards has the lowest AIC (decisively) and recovers its
true parameters.

## Regenerating the data

`growth.exp` is the Richards model's output at the truth (`r=0.8, K=100, b=3`) plus
seeded gaussian noise, regenerated from [`_manifest.py`](../_manifest.py) by:

```bash
python examples/tutorial/regenerate_data.py 45_model_selection
```

## Notes

- `richards.bngl` and `von_bertalanffy.bngl` are the analytical-ODE catalog's
  `richards_growth` and `von_bertalanffy_growth`, cut to clean edition-2 models (the
  `Clock` and `Analytical_*` reporting functions removed) and renamed to observe
  `Obs_N` so all four candidates share the data column. `logistic`/`gompertz` are the
  lesson-01/03 laws.
- AIC is computed here from the chi-square with a known per-point `_SD`. With an
  *estimated* noise level you would use the log-likelihood form
  (`AIC = 2k − 2·ln L`) and count the noise parameter in `k`.
