# Lesson 44 — Where the search starts (initialization)

**Feature:** `initialization` / `initialization_distribution` · **Difficulty:** ★★ (recovery tier)

Before a population optimizer takes its first step, it has to **draw an initial
population**. Two config keys control that draw:

| Key | Controls | Values |
| --- | --- | --- |
| `initialization` | **how** the points are spread | `lh` (Latin hypercube, space-filling — the default) · `rand` (independent uniform draws) |
| `initialization_distribution` | **where** they are drawn from | `prior` (each parameter's prior distribution — the default) · `bounds` (uniform within finite bounds) |

## `initialization` — how the points are spread

`lh` (the default) draws a **Latin hypercube**: it splits each parameter's range
into as many strata as there are population members and guarantees one member per
stratum, so the population **covers** the space evenly. `rand` draws each member
independently, which can leave gaps and clumps. On any single run either can get
lucky, but LH's uniform coverage makes it the more reliable default — the payoff
grows with the population size and the number of parameters.

## `initialization_distribution` — where they come from, and why it matters

This lesson's two confs, on lesson 07's hard multimodal oscillator with a
**deliberately tiny budget** (10 chains, 8 iterations, no refine), show that on a
rugged landscape it is *where* the population starts that decides whether the fit
finds the basin at all:

- **[`prior_seeded.conf`](prior_seeded.conf)** puts an **informative** `normal_var`
  prior near the truth on each rate. Because `initialization_distribution = prior`
  (the default) draws the initial population *from those priors*, the search is
  seeded right where you believe the answer is — and the tiny budget lands on
  `(alpha, gamma) = (1.2, 0.8)`.
- **[`uninformed.conf`](uninformed.conf)** uses flat `uniform_var` priors over the
  whole box. With no informative prior to draw from, the population scatters
  uniformly and eight iterations can't climb out of a wrong-frequency local minimum
  — the fit lands **far** off.

```bash
pybnf -c prior_seeded.conf   # informative prior seeds the search -> recovers
pybnf -c uninformed.conf     # flat priors, same tiny budget       -> dragged off
```

The prior only **seeds** the optimizer here — the data (the sum-of-squares
objective) is still what the fit minimizes, so a *wrong* prior would only cost you
the head start, not bias the answer. (For a Bayesian **sampler** the prior also
shapes the posterior — lessons [15](../15_petab_priors)/[27](../27_priors)/[32](../32_prior_gallery).)

`initialization_distribution = bounds` draws uniformly within each parameter's
**finite** bounds, so it needs bounded priors (`uniform_var` / `loguniform_var`);
an unbounded `normal_var` has no finite bounds to draw from, which is why `prior`
is the default.

## The catch, and the fix

The tiny budget is a magnifying glass: it isolates the *initialization* effect.
Give `uninformed.conf` a real budget and a `refine` polish — as the lesson-07
bake-off does — and it recovers too. The lesson is not "flat priors don't work,"
it's: **a good starting distribution buys you convergence** — cheaper fits, or a
fighting chance when the budget is tight and the landscape is hard.

## The test

Both confs are verified by the manifest-driven recovery suite
([`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py),
`-m recovery`): `prior_seeded.conf` recovers `(alpha, gamma)` within 5% from the
tiny budget, and `uninformed.conf` must miss by ≥10% (so the contrast is not
vacuous).

## Notes

- The model is lesson 07's linearized Lotka-Volterra oscillator, reused because its
  multimodal `sqrt(alpha*gamma)` landscape is exactly where a good starting
  population earns its keep. `oscillator.exp` is regenerated from
  [`_manifest.py`](../_manifest.py) by
  `python examples/tutorial/regenerate_data.py 44_initialization`.
