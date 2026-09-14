# A differential evolution candidate always changes a parameter its donors move, under the modern edition, because an exact copy of its base ties the base under the seed policy and collapses the population onto one parameter set (issue #698)

## Status

Accepted and implemented (2026-09-14). The fault was found by ADR-0142 while measuring the
learned mutation settings (#667) on the stochastic recovery benchmark (ADR-0140), and left
out of that change's scope. The edition gating follows ADR-0137.

## The problem

Differential evolution builds a candidate from a base parameter set p1 and two donors p2
and p3 (four under a `2` strategy): each parameter of p1 is moved by `mutation_factor`
times the donors' difference with probability `mutation_rate`, and kept otherwise. Nothing
guaranteed that any parameter moved. At the default rate the candidate is an exact copy of
p1 one time in eight on three parameters, one in sixteen on four, one in sixty-four on six.

Under the default seed policy (`stochastic_seed = auto`) a trajectory's seed is hashed from
the parameter values, so a copy runs the same simulations as p1 and scores exactly what p1
scored. It competes for the slot the strategy names, which under `rand` and `best` is not
p1's own, and takes it whenever p1 is the better of the two, without having searched
anything. In `ade`, which proposes from the population as it stands after every result,
the slot a copy took is a base for the very next proposal, so copies build copies. Once
every member is the same parameter set, every donor difference is zero, nothing further can
happen, and the convergence test sees a spread of zero and stops the run, reporting
convergence. On the benchmark, plain `ade` stopped on its own in 24 of its 30 fits, after
3,288 to 19,209 of 20,000 simulations (4,000 for the network-free problem).

Binomial crossover's answer is one parameter chosen in advance that is always taken from the
mutant, and the learned settings added it (ADR-0142). It is not enough in PyBNF, for a
reason that is particular to how PyBNF crosses. Published differential evolution crosses the
mutant with the member the candidate will replace, so an unmutated value stays where it was.
PyBNF crosses it with p1, and under `rand` and `best` the candidate replaces a different
member, so after a replacement two members hold p1's unmutated values. Members come to share
values, and the donors then often agree on the parameter chosen in advance, where their
difference is zero. A candidate built there is a copy after all: the learned-settings runs,
which had the guarantee, still stopped early in 4 of their 30 fits.

## The decision

### The guarantee

`DifferentialEvolutionBase._forced_parameter` chooses the parameter a candidate always
mutates, after the picks and the learned settings' draw and before the coins:

1. One parameter is drawn at random, the draw binomial crossover makes.
2. If the donors' difference does not change its value, the choice is made again among the
   parameters whose value the difference does change, each equally likely. The draw is
   taken from the parameters with a nonzero difference, dropping any whose value it still
   leaves as it was, so the result is uniform over the parameters that move, and the whole
   procedure (step 1, then step 2 when needed) picks each of them with the same
   probability.
3. If the difference changes no parameter, because the donors are one parameter set or,
   under a `2` strategy, their two differences cancel, other donors are drawn from the
   members other than the base, up to as many times as the population has members, and
   step 2 is applied to each.
4. If none of those changes anything, the candidate is left a copy of its base.

A move is judged on the value, not on the difference: a nonzero difference too small to
change a value in floating point is not a move. Step 4 is possible only in a population that
has all but collapsed. Under a `1` strategy, when a single member other than the base
differs from the rest, a donor pair drawn at random includes it with probability 2/(N - 1)
in a population of N, so all N draws miss it 11 percent of the time at a population of 20
(approaching e^-2, 14 percent, as N grows); with two such members 1 percent, with three
0.1 percent. A population whose members other than the base are all one parameter set gives
up every time, which is correct: no difference exists to search with.

When the first parameter drawn moves, which every parameter does until members share
values, the procedure is exactly one draw, the one the learned settings made before, and it
reuses the moved parameter it built to test the move. The common case therefore costs one
donor difference more than the legacy proposal and builds no extra parameter; the fallback
computes every parameter's difference and builds one moved parameter per parameter tried.

### On under the modern edition, off under the legacy one

The guarantee changes the search operator, and so every parameter set a fit queues after its
first shared value. ADR-0031's contract is that a configuration naming no edition keeps
behaving as it always has, so, as for ADR-0137's diverse reference set, the guarantee is on
under `edition = 2` and above and off under the legacy edition, where the proposal makes
exactly the draws it made before. `de_force_mutation` sets it explicitly either way, which is
also what makes it measurable on the benchmark, whose configurations are legacy syntax.

### The learned settings use the same code, always

A learned rate can sit near 0, where most candidates would be copies of their base, and a
copy can never be a success, so `de_adapt_mutation = 1` turns the guarantee on whatever the
edition. The learned settings' own forced draw is gone; they call the same method. An
explicit `de_force_mutation = 0` next to `de_adapt_mutation = 1` is overruled, with a note
printed and logged rather than silently.

## Why this form of the guarantee

* **Draw the parameter over all of them, as published, and repair only an exact copy.**
  Considered first, and it keeps the letter of the guarantee. But once members share values
  the parameter drawn is often one the donors agree on, and whether the candidate differs
  from its base then rests on the coins; the repair fires only when every coin also missed.
  Drawing again among the parameters that move keeps the intent as well: the parameter the
  candidate is sure to change is one it changes. Both make the same first draw, so they
  agree until the first parameter drawn fails to move. And the repair cannot do better on
  the shared values described below: in every case it changes at most as many parameters as
  the decision does, so a candidate it builds inherits at least as many of its base's values.
* **Draw the parameter among the moving ones from the start.** The same distribution as
  the decision, but a different draw from the learned settings' whenever any parameter fails
  to move, not only the one drawn, so a learned-settings run would part from its earlier
  behaviour as soon as any parameter failed to move rather than only when the one drawn does.
* **Perturb a copy by a small random step,** as `de` does to a candidate that duplicates
  one already in flight. That step has nothing to do with the population's spread, it is not
  the operator, and it would hide a population that has genuinely collapsed from the
  convergence test.
* **Draw donors until something moves, without a bound.** A population of one parameter set
  never ends the loop, and a floating-point corner can leave a population with differences
  that never change a value. The bound is the population size, for the reason given above.
* **Cross the mutant with the member it will replace**, as published, which stops values
  being shared at the source. That changes the operator under every strategy and every
  setting; it is the larger defect's fix, not this one's (see the next section).

## What this leaves

The guarantee stops a candidate from being a whole copy of its base. It cannot move a
parameter whose value every member shares, since no donor difference is nonzero there, and
that state has the same cause: PyBNF crosses the mutant with p1 rather than with the member
the candidate replaces, so p1's unmutated values spread through the population. Under the
`rand` strategies the guarantee makes it rare (no `rand1` or `rand2` run on the noisy bowl
below ended with such a parameter). Under `best` it does not, because every candidate is
built from the best member and inherits its unmutated values: with the guarantee on, 9 of the
10 `best1` runs on the deterministic bowl still ended with at least one parameter no member
could move, and 6 stopped early, 4 of them with the whole population at a true objective of
0.01 or more. Crossing with the member the candidate replaces, as published differential
evolution does, would end the sharing at its source. It changes the operator under every
strategy and would change what the learned settings should judge a success against
(ADR-0142), so it is left for its own issue, #700.

## The evidence

### An analytical bowl, with and without seed-derived noise

A three-parameter bowl, the sum of (x - 1)^2 over [-10, 10]^3, driven through each method's
own `got_result` for 4,000 evaluations: population 10, the default rate and factor of 0.5,
`stop_tolerance = 0` so that only a population of identical objective values stops a run,
five seeds per cell. In the noisy rows each evaluation adds 0.5 times a standard normal drawn
from a seed hashed from the parameter values, which is how `stochastic_seed = auto` seeds a
simulation: an exact copy scores exactly what its base scored and anything else draws fresh
noise. "Early" counts runs that stopped before 4,000 evaluations; "best" is the median, over
the seeds, of the smallest true (noise-free) objective in the final population.

| method | strategy | noise | early, legacy | early, guarantee | best, legacy | best, guarantee |
|---|---|---|---:|---:|---:|---:|
| `ade` | `rand1` | yes | 5 | 0 | 0.074 | 0.018 |
| `ade` | `rand1` | no | 2 | 0 | 0.071 | 0.00063 |
| `ade` | `rand2` | yes | 5 | 0 | 0.22 | 0.043 |
| `ade` | `best1` | yes | 5 | 1 | 1.1 | 0.21 |
| `ade` | `all1` | yes | 0 | 0 | 0.057 | 0.031 |
| `de` | `rand1` | yes | 1 | 0 | 0.84 | 0.019 |
| `de` | `rand1` | no | 0 | 0 | 1.1 | 0.0014 |
| `de` | `rand2` | yes | 2 | 0 | 0.18 | 0.020 |
| `de` | `best1` | yes | 0 | 0 | 3.2 | 0.87 |
| `de` | `all1` | yes | 0 | 0 | 0.020 | 0.039 |

Every legacy `ade` run under `rand1` with noise ended as a single parameter set; with the
guarantee the final populations held 5 to 10 distinct members. The deterministic `rand1`
rows show that the copies cost more than early stops: no legacy `de` run stopped early, yet
the legacy populations of both methods ended two to three orders of magnitude short of the
guarantee's objective, and 3 of the 5 legacy `ade` runs ended with a parameter every member
shared. Under `all`, where the base is the slot and no value is shared, the guarantee changes
nothing that five seeds can see, as it should not.

How often each branch of the guarantee ran, over all proposals of the guarantee runs:
under `rand1` with noise the first parameter drawn failed to move in 17 percent of `ade`
proposals and 14 percent of `de`'s, the donors were drawn again in 5.5 and 4.0 percent, and
no proposal gave up; without noise the first draw failed in 6 and 8 percent and the donors
were drawn again in fewer than one proposal in a thousand. Under `best1`, where values are
shared fastest, the first draw failed in 29 to 43 percent, and `ade` gave up on 0.5 to 1.6
percent of proposals, in populations that had all but collapsed.

The learned settings before and after they share the guarantee, on the same bowl (the
previous code's own forced draw against this method; both methods, all four strategies,
with and without noise, five seeds, so 40 runs each way with noise and 40 without): with
noise, the previous code stopped early in 13 of its 40 runs, every one a population
collapsed onto a single parameter set with a true objective between 0.08 and 1.7, and this
code in none. Its median true objective was lower in four of the six noisy cells where the two
differ (the `all` cells are identical, as they must be) and higher for `de` under `best1` and
`rand2`. Without noise the comparison
is less clean. Most early stops under both are populations converged exactly onto the
optimum, a true objective of 1e-14 or less, which is the convergence the test is there to
detect; but runs that stopped early away from the optimum rose from 2 to 7 of 40. Each of
the seven was a population of ten distinct members with one or two parameters every member
shared, the defect described above. The learned rate is not what separates them: it ended
between 0.13 and 0.73 in those seven runs and between 0.18 and 0.82 in comparable runs that
did not stall.

### A two-parameter Rosenbrock valley

Two existing end-to-end tests fit the banana target with a = 1 and b = 20 over [-5, 5]^2
under edition 2 with `de`, population 20, 300 generations and the default convergence stop,
from seed 42, and both failed once the guarantee became edition 2's default: the population
converged on the valley floor at x1 = 1.78. From seeds 1 to 40, with the tests' own
criterion for success (both parameters within 0.1 of the mode and a best objective of 0.05
or less), `de` succeeded from 22 seeds without the guarantee and 37 with it (17 seeds only
with it, 2 only without; two-sided sign test p = 0.0007), and `ade` from 19 and 38 (19 and
0; p = 0.000004). Seed 42 is one of the four of 41 from which `de` with the guarantee
converges prematurely, and neither a stricter convergence stop nor twice the generations
changed any of the four. With 30 members instead of 20, `de` with the guarantee reached the
mode from all 41 seeds (34 without it), and the two tests now use 30. Under `all1`, where no
values are shared, both operators reached it from all 41 at 20 members, which is what the
previous section's account of shared values predicts, though `all1` differs in more than that.

### The stochastic recovery benchmark

Six problems, five fit seeds each, the frozen budgets, from the same seeds as ADR-0142's
measurement (`benchmarks/stochastic_recovery/results/de_force_mutation_5seeds.json`; the rows
compared against are in `results/de_adapt_5seeds.json`, and the README has the per-problem
table). Success is every identifiable parameter within a factor of two of the truth; the
p values are two-sided sign tests over seeds.

None of the 120 fits stopped before its budget, against 24 of 30 for plain `ade` and 4 of 30
for the learned settings as ADR-0142 measured them. `ade` with the guarantee succeeded from 12
seeds of 30 against 4 for plain `ade`: 8 seeds only it won and none only plain `ade` did
(p = 0.008), with the final error lower on 19 seeds and higher on 11. That is level with
ADR-0142's control, which always mutated one parameter drawn at random but did not draw again
when the donors could not move it (10 of 30; 6 seeds against 4). `de`, which never collapsed,
went from 6 successes to 8 (5 seeds against 3, final error lower on 15 and higher on 15),
nothing five seeds resolve. The learned settings, sharing the guarantee, no longer stopped
early, and added nothing measurable over the guarantee alone: 9 successes against 12 under
`ade`, 8 against 8 under `de`. Under `de` they fell from ADR-0142's 11 to 8 (one seed against
four, p = 0.38), which five seeds cannot tell from noise, but it means that measurement's gain
under `de` is not reproduced once its code path changes. `de_adapt_mutation` stays off by
default, as it was.

## Consequences

* `DEFamilyConfig` gains `de_force_mutation` (unset resolves by edition), registered in the
  parse layer, the docs and the effective-config golden; `de` and `ade` both own it.
* `DifferentialEvolutionBase.new_individual` reads the guarantee from `force_mutation`,
  resolved once in `__init__`; the donor difference, the move test and the choice are
  `_difference`, `_moved` and `_forced_parameter`.
* The legacy proposal is byte-identical: the fake-rng oracles in `tests/test_diff_evolution.py`
  pin its draws, and proposal logs of `de` and `ade` under three strategies, with and without
  seed-derived noise, matched the previous code's value for value.
* A learned-settings run is identical to its earlier behaviour until the first parameter
  drawn fails to move, and differs from then on. ADR-0142's `de_adapt` and `ade_adapt` rows
  were measured before this change; the rows measured after it are in the new results file.
* The two end-to-end tests that fit the Rosenbrock valley under edition 2
  (`test_inline_banana_de_recovers_mode`, `test_expression_de_recovers_rosenbrock_mode`)
  use 30 members instead of 20, for the reason measured above; their seed is unchanged.
* Three tutorial lessons fit with edition 2's `de`, and their recovery checks (seed 1234)
  missed with the guarantee on, although over seeds 1 to 20 each recovered more often with
  it than without. Their budgets are raised so the check's seed recovers for a reason rather
  than by luck. Lesson 44 runs 10 iterations instead of 8 in both confs: `prior_seeded.conf`
  then recovers from 20 of 21 seeds counting 1234 (at 8, 19 of 20 with the guarantee and 11
  without), and `uninformed.conf` is still dragged off from 15 of 21, 1234 among them.
  Lesson 24 runs 60 instead of 30, and `moment_fit.conf` recovers from 19 of 21 (at 30, 5 of
  20 with the guarantee and 2 without). Lesson 25 runs 80 instead of 40. Its fit has a
  second basin, near k_transit = 10.1 and k_abs = 14.7, which the guarantee's runs entered
  from 11 of 21 seeds at 80 iterations and 10 of 21 at 120; the budget does not change that. What the
  budget changes is convergence in the documented basin: at 40 the check's seed was there but
  3.6 percent short of the 3 percent tolerance, and at 80 every run in that basin converged.
