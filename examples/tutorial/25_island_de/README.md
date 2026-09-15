# Lesson 25 — Island-based Differential Evolution

**Feature:** `islands` / `migrate_every` / `num_to_migrate` on `job_type = de` · **Difficulty:** ★★★

Plain Differential Evolution (lesson 07) keeps a single population. On a rugged or
high-dimensional error surface that one population can **converge prematurely** — the
whole swarm slides into a single basin before it has explored the others, and the fit
gets stuck in a local optimum. **Island DE** is the standard remedy.

This lesson's fit also shows where any such remedy stops: an error surface with two minima
that fit the data almost exactly as well as each other.

## How islands work

The population is split into several semi-isolated sub-populations — *islands* — each
running its own DE. Every so often the islands **migrate**: each sends a few of its
individuals to the others. The islands explore different regions in parallel, and
migration spreads good solutions between them without homogenizing the search too
quickly. It is the same `job_type = de`, enabled with three keys:

```
population_size = 40      # split across the islands ...
islands         = 4       # ... into 4 sub-populations (10 individuals each)
migrate_every   = 10      # every 10 iterations, each island ...
num_to_migrate  = 2       # ... sends 2 individuals to the others
```

`population_size` is divided across the islands, so `40 / 4 = 10` individuals per
island (each island needs at least 3). With `islands = 1` (the default) there is a
single population and no migration — ordinary DE.

## The test problem

A transit-compartment pharmacokinetic model — an oral dose moves through two transit
compartments (rate `k_transit`), is absorbed into plasma (`k_abs`), and eliminated
(`k_elim`):

```
Transit1 → Transit2 → Absorption → Central → (eliminated)
             k_transit   k_transit    k_abs      k_elim
```

We measure **only the central (plasma) compartment** and fit all three rates to that
single curve:

```bash
pybnf -c island_de.conf
```

Every run ends on a curve that matches the data, with `k_elim ≈ 0.96`. For the other two
rates it reports one of **two** answers, and which one depends on the run's random seed:

| | `k_transit` | `k_abs` | `k_elim` | best objective |
| --- | ---: | ---: | ---: | ---: |
| the rates the data were made from | 12.76 | 9.11 | 0.96 | 1e-9 or less |
| a second minimum | 10.09 | 14.65 | 0.96 | 2.2e-8 |

The true rates come back in about two runs out of five. Neither answer is a failed fit: the
objective has these two minima, and the fit converges to one of them. It gets all the way
there because the conf finishes with a gradient polish, `refine_method = trf`. DE can stall
in the long, narrow valley the two minima lie in, and the default Simplex polish can stop
partway along it.

## One curve, two answers

Set elimination aside, and what shapes the plasma curve is how long the dose takes to
reach plasma: two transit steps at `k_transit` and one absorption step at `k_abs`. That
delay has mean `2/k_transit + 1/k_abs` and variance `2/k_transit² + 1/k_abs²`, and the
curve fixes both closely. Two equations in two rates, but not one solution: eliminating
`k_abs` leaves a quadratic in `1/k_transit`, whose roots are the true rates and
`k_transit = 10.07`, `k_abs = 14.73`, within 0.5% of the second minimum. Only the third
and higher moments of the delay tell the two pairs apart:

| `k_transit`, `k_abs` | mean delay (h) | variance (h²) | third central moment (h³) |
| --- | ---: | ---: | ---: |
| 12.76, 9.11 | 0.26651 | 0.024333 | 0.0045706 |
| 10.09, 14.65 | 0.26648 | 0.024304 | 0.0045300 |

So the second pair's plasma curve never strays more than 0.0002 from the first, on a peak
of 1.7. The true rates score lower only because these data are noise-free. Add measurement
noise with a standard deviation of 0.001, less than 0.1% of the peak, and the chi-square
values of the two minima differ by a few hundredths, far less than one: no fit could say
which pair made the data.

## Why the search lands in either

A population settles into one basin long before it can see which basin is lower. In this
fit most members sit in one basin by around generation 35, while the best objective is
still ten to a thousand times the 2.2e-8 that separates the minima, and that basin is
almost always where the fit ends. Migration every 10 generations carries it from island to
island, so the islands do not explore separately for long, and a single population of 40
does no better.

Keeping the islands apart for longer raises the odds, but only if each island gets close
to its own minimum before it shares, and even then some seeds end in the second one: every
island is a separate draw between two basins of about the same size. And an answer that
hangs on a difference of 2e-8 is not one to rely on anyway. With real data that difference
is noise.

## What would tell them apart

Data that sees absorption apart from transit. Add a second experiment in which the dose is
placed straight into the absorption compartment: its plasma curve depends on `k_abs` and
`k_elim` alone. Fitted jointly with the first curve (lesson 16 shows how), the objective
has a single minimum, at the true rates.

And a habit worth keeping for any global fit: **run it more than once, from different
seeds**. Two runs that end at different parameters with about equally good objectives mean
the data do not pin those parameters down, whichever optimizer produced them. Lesson 2's
profile likelihood asks the same question systematically.

## When to reach for it

- **Many parameters** (roughly ≥ 5) or a landscape you know is multi-modal.
- A plain DE run that keeps landing in **different** local optima on repeated runs —
  a sign the single population is under-exploring. Compare their objectives first: if
  the optima fit about equally well, as they do here, the data are the limit, not the
  search.

Islands cost nothing extra per evaluation (the population is the same size, just
partitioned), so they are a cheap first thing to try when a DE fit looks unreliable.
Tune `migrate_every` (larger = more independent islands, slower sharing) and
`num_to_migrate` (larger = faster homogenization) to trade exploration against
convergence speed.

## The data & test

`transit_pk.exp` is regenerated by the shared `regenerate_data.py` (the plasma curve
at the true rates). The expected result is recorded in
[`_manifest.py`](../_manifest.py) and exercised by
[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py) from three
random seeds: each fit must recover `k_elim` to within 3%, and `k_transit` and `k_abs` to
within 3% of either minimum.
