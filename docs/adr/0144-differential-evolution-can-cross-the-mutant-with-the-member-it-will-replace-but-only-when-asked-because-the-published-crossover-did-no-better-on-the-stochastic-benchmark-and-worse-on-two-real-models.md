# Differential evolution can cross the mutant with the member it will replace, but only when asked, because the published crossover removed the base crossover's frozen parameters on analytical targets yet did no better on the stochastic benchmark and worse on two real models (issue #700)

## Status

Accepted and implemented (2026-09-15), off by default under every edition. Found while
measuring ADR-0143 (#698), whose copy guarantee could not reach it. The plan was to make it the
default under `edition = 2`, as ADR-0137 and ADR-0143 did for their changes; the measurements below
did not support that, and the decision to ship it as an opt-in key was taken on them.

## The problem

Differential evolution builds a candidate from a base p1 and two donors (four under a `2`
strategy): each parameter is mutated to p1 plus `mutation_factor` times the donors'
difference with probability `mutation_rate`, and otherwise keeps a value from the member the
mutant is crossed with. Published differential evolution (Storn and Price 1997, and SHADE
after it) crosses with the member the candidate will replace, its target, so an unmutated
value is the target's own and stays where it was. PyBNF crossed with p1. Under the `all`
strategies p1 is the target and the two agree. Under `rand` and `best` they do not: a
candidate that wins its slot holds p1's unmutated values while p1 stays in the population,
so after every such replacement two members share values. Values spread, and once every
member holds the same value of a parameter, every donor difference is zero there and that
parameter can never move again. The other parameters converge around it and the fit stalls
short of the optimum, and the convergence test can stop it there. Under `best`, where every
candidate is built from the best member, the spread is fastest.

ADR-0143's guarantee, that a candidate always changes a parameter its donors move, removed
whole copies of the base and made shared values rarer under `rand`, but it cannot move a
parameter every member already shares, and under `best` it did not stop them forming: on a
noise-free three-parameter bowl, 9 of 10 `best1` runs with the guarantee still ended with
such a parameter.

## The decision

### An opt-in crossover

`new_individual` takes the index of the member the candidate will compete with,
`target_index`, and both methods pass it: `de` the slot `jj` it is filling, `ade` the slot `j`
whose result just arrived. With `de_cross_with_target` on, an unmutated parameter keeps that
member's value; a mutated one is p1 plus the scaled difference, as before. Without a target
index, or with the key off, the mutant is crossed with p1 as it always was. The picks are
unchanged: p1 and the donors are drawn from the whole population, the target included (see
Alternatives).

`de_cross_with_target` is a plain switch, default 0 under every edition, read once in
`__init__` like `de_adapt_mutation`. With it off, every proposal is byte-identical to ADR-0143's,
with and without the copy guarantee and the learned settings.

### The copy guarantee, stated against both members

ADR-0143's guarantee moves one parameter by the donors' difference. Crossed with the target it
must rule out two copies, not one: the moved value must differ from the base's value, so a
zero difference is never a move, and from the target's value, so the moved value does not
land on what the candidate would keep anyway. The candidate then differs from its base and
from its target in the parameter moved. Crossed with p1 the two tests are one, and the rules
are ADR-0143's exactly, draw for draw.

The first test is the one that is easy to drop, and dropping it was measured. A first version
of this change counted a zero difference as a move wherever p1 and the target differ, since
the mutant then keeps p1's value, which is not the target's. That lets two identical donors
through, and when every mutated parameter has a zero difference the candidate is its base
exactly, which is #698's copy again, reached from the other side: it ties the base's score
under the seed policy, wins the target's slot, and the next pair of identical donors is more
likely. On the noisy bowl below that version proposed 126 exact copies of the base over 30
runs, and the only four runs that stopped early were runs with such copies. With a zero
difference never a move it proposed none, and no run stopped early. Copies of some other member
remain as coincidences, 48 in 120,000 proposals, none of them followed by a stall.

### The learned settings, judged against the member crossed with

A success is a candidate that beats the member its mutant was crossed with, because that
member is what the settings were applied to. Crossed with the target, that is SHADE's own
rule: success against the member the candidate competes with. ADR-0142 judged against p1
only because, crossed with p1, a candidate that merely copied a better base would win against
its slot without its settings having done anything; crossed with the target that copy does
not arise. `_trial_settings` records the reference fitness of whichever member was crossed.

## Why it is off by default

Crossing with the target did what it was built to do wherever the landscape was synthetic: no
frozen parameters and exact convergence on a separable bowl, better results on a correlated,
ill-conditioned one, and every seed reaching the mode of a narrow curved valley. It also helped the
learned settings on the stochastic recovery benchmark. But on the problems PyBNF's users fit it
did not: on the benchmark with fixed settings it was level with crossing with the base over fresh
seeds and worse on one problem over twenty, and on two of the tutorial's ODE models it was worse,
one ending further from the documented values on 17 of 21 seeds and the other converging more
often into a second, slightly worse basin. A default should improve the fits it changes, and this
one would have changed every `rand` and `best` fit under `edition = 2` for a gain the real models
did not show. The defect it removes is worst under `best`, which is not the default strategy, and
a user who sees parameters that every member of the population shares can turn it on.

Why the published crossover does worse on those two models is not established here. It is not
correlation between parameters as such: it did better on the correlated bowl. Crossing with the
base copies good values from member to member, which may help a population commit to the better
basin early, but that was not measured.

## Alternatives

* **On under `edition = 2`,** the plan, following ADR-0137 and ADR-0143. Rejected on the evidence
  above.
* **On under `edition = 2` for the `best` strategies only,** where freezing is worst and the
  analytical gains were largest. No real model was measured under `best`, so the same risk stands
  unmeasured. Not adopted.
* **Exclude the target from the picks,** as published differential evolution draws r1, r2 and r3
  distinct from the target. A proposal built on the target itself is the `current` form of the
  operator and harmless, and excluding it would raise the minimum population from 3 to 4 (6 under
  a `2` strategy) and change every proposal's draws. Not adopted.
* **Also rule out a copy of any other member.** A candidate that equals a third member ties that
  member's score too, but it takes a coincidence: the same base, donors and mutated parameters as
  that member was built from, on a target that shares its other values. It happened 48 times in
  120,000 proposals on the noisy bowl, with no stall after any of them. Not adopted.
* **Keep judging the learned settings against p1 when crossing with the target.** Inconsistent
  once the settings are applied to the target, and SHADE's rule is the target. Not adopted.

## The evidence

### An analytical bowl

The bowl of ADR-0143, the sum of (x - 1)^2 over [-10, 10]^3, driven through `got_result` for
4,000 evaluations, population 10, rate and factor 0.5, `stop_tolerance = 0`, both methods and
all four strategies, with and without noise seeded from the parameter values, five seeds per
case; "#698" is ADR-0143's guarantee crossed with p1, "#700" the same guarantee crossed with
the target. A stall is a run that stopped early or ended with a parameter every member
shared, while the population's best true objective was 1e-6 or more.

| | #698 | #700 |
|---|---:|---:|
| stalls, no noise, fixed settings (of 40) | 8 | 0 |
| stalls, no noise, learned settings (of 40) | 11 | 0 |
| stalls, noise, fixed settings (of 40) | 5 | 0 |
| stalls, noise, learned settings (of 40) | 1 | 0 |

Without noise every `rand1`, `best1` and `rand2` run crossed with the target, fixed or
learned, reached the optimum exactly, where #698 ended at medians of 6e-4 (`ade`, `rand1`),
1.4e-3 (`de`, `rand1`), 4e-4 and 0.44 (`best1`). With noise, crossing with the target cut the
median true objective under `best1` from 0.21 to 0.013 (`ade`) and from 0.87 to 0.12 (`de`),
and with the learned settings lowered it in all six noisy cells that differ, by factors of 1.3
(`de`, `rand2`) to 66 (`de`, `best1`). With fixed settings under `rand` and noise it did not
help: over the 20 seeds of those four cells it ended lower than #698 on 7 and higher on 13 (sign
test p = 0.26), with a median of 0.035 against 0.018 for `ade` under `rand1`, although #698's
worst runs there, 0.30, 0.42 and 0.20, have no counterpart, the worst crossed with the target
being 0.07. The fallback
all but disappears: under `rand1` with noise the first parameter drawn failed to move in 0.6
percent of proposals, against 17 and 14 percent (`ade`, `de`) crossed with p1.

The driven test in `tests/test_diff_evolution.py` is one of these cases: `de` under `best1`
without noise, from seeds 1 to 10, never got below a true objective of 0.012 crossed with p1
(up to 24, and with a frozen parameter in 6 of the 10 runs), and reached the optimum from every
seed crossed with the target, to 7e-29 at worst.

### A correlated, ill-conditioned bowl

The same comparison on 0.5 (x - 1)^T Sigma^-1 (x - 1), with Sigma a fixed random rotation of the
variances 100, 1 and 0.01 (condition number 10^4), `rand1` and `best1`, both methods, with and
without noise, five seeds. Crossing with the target ended lower than crossing with the base on 34
of the 40 seed-by-seed comparisons with fixed settings and 31 of 40 with learned ones; with fixed
settings and no noise its medians were 1e-5 or less where crossing with the base ended between
0.006 and 3. So correlation between parameters does not by itself favour crossing with the base.

### A two-parameter Rosenbrock valley

The fit of ADR-0143's two end-to-end tests (edition 2, banana with a = 1 and b = 20, 300
generations, the default stop) at population 20, from seeds 1 to 40 and 42: `de` with the
guarantee reached the mode from 37 seeds crossed with the base and from all 41 crossed with the
target (4 seeds only with the target, none only without); `ade` from 39 and 41. The tests keep
ADR-0143's 30 members, since the default crossover is still the base.

### The stochastic recovery benchmark

Five seeds on all six problems (`benchmarks/stochastic_recovery/results/de_cross_target_5seeds.json`,
paired with `results/de_force_mutation_5seeds.json`): with fixed settings `ade` succeeded from 7
seeds of 30 crossed with the target against 12 crossed with the base (1 seed only with the target,
6 only with the base; two-sided sign test p = 0.12), `de` from 6 against 8; no fit stopped early.
With the learned settings, 11 against 9 (`ade`) and 10 against 8 (`de`, final error lower on 20 of
30 seeds, p = 0.10).

The fixed-settings drop was then tested over twenty seeds on the three problems where a method
succeeds often enough to tell variants apart (`results/de_cross_target_seeds6to20.json`), both
crossovers run on seeds 6 to 20. Over all twenty seeds `ade` succeeded from 31 of 60 crossed with
the target against 39 crossed with the base (8 / 16, p = 0.15), with the final error higher on 38
seeds and lower on 22 (p = 0.05); on the fifteen seeds that had not been used to choose the
problems, 25 against 27 (8 / 10, p = 0.82). `de` succeeded from 31 of 60 against 28 (12 / 9), and
on the fresh seeds from 26 against 20. The one consistent difference was McKane_PhysRevLett2005 under
`ade`: 3 successes of 20 crossed with the target against 11 crossed with the base (p = 0.02), 3
against 7 on the fresh seeds.

### Three ODE models from the tutorial

The tutorial's recovery checks run each lesson's conf from one seed. Lessons 07, 24 and 25 fit
with `de` or `ade` under `edition = 2`; each was run from seeds 1 to 20 and 1234 with the new
crossover on and off, at the lesson's committed budget.

* Lesson 07 (`oscillator_de.conf`, `oscillator_ade.conf`, a linearized Lotka-Volterra oscillator):
  both crossovers recovered the documented values from all 21 seeds.
* Lesson 24 (`moment_fit.conf`, the moments of a birth-death-immigration process): crossed with the
  target it recovered the three rates within 3 percent from 15 seeds, against 19 crossed with the
  base (2 / 6, p = 0.29), and ended further from them on 17 of the 21 seeds (p = 0.007), a median
  worst error of 2.1 percent against 0.3.
* Lesson 25 (`island_de.conf`, a transit-compartment model observed only in the central
  compartment): its objective has a second basin near k_transit = 10.1 and k_abs = 14.7, whose
  best objective (2e-8 or more) is slightly worse than the documented basin's (6e-19). Crossed with
  the target the fit converged into it from 15 seeds of 21, against 11 crossed with the base, and
  recovered the documented values from 6 against 10. From the recovery check's own seed, 1234, it
  converged into the second basin, where crossing with the base does not.

## Consequences

* `DEFamilyConfig` gains `de_cross_with_target` (default 0), registered in the parse layer, the docs
  and the effective-config golden; `de` and `ade` both own it.
* `new_individual` gains `target_index`, and both methods pass the slot a candidate competes for;
  with the key off it is ignored. The copy guarantee's `_moved` compares a moved value against the
  base's and the crossed member's, which with the key off is ADR-0143's test; `_trial_settings`
  records the crossed member's fitness.
* With the key off the proposal is byte-identical to ADR-0143's: proposal logs of `de` and `ade`
  under `rand1`, `best2` and `all1`, with and without seed-derived noise, and with the guarantee
  and the learned settings, matched the previous code value for value in all 36 comparisons.
* No test or tutorial budget changes: the default crossover is unchanged.
