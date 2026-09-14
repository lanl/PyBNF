# Differential evolution learns its mutation settings from a success history judged against the candidate's base, off by default, because the two settings cannot be chosen in advance and judging success against the slot teaches the run to clone (issue #667)

## Status

Accepted and implemented (2026-09-14). The mechanism is SHADE's success-history parameter
adaptation (Tanabe and Fukunaga 2013), adapted to how PyBNF builds a candidate. The evidence
is in this document and in the stochastic recovery benchmark (`benchmarks/stochastic_recovery/`,
ADR-0140).

## The problem

Differential evolution builds each candidate by taking a parameter set and shifting some of its
values by a scaled difference between two other members of the population. Two settings decide
that: `mutation_rate`, how often a value is changed, and `mutation_factor`, by how much. Both are
read once at startup and used from the first iteration to the last. Neither has a value that is
right for every model, because the right value depends on the model, on whether its parameters
act independently or together, and on whether the search is exploring or refining. A user who
does not know what to set is left guessing, and nothing in the documentation can tell them.

The SHADE family of differential evolution methods learns both values during the run. It keeps a
short memory of the settings that produced a successful candidate, draws each new candidate's
settings around a remembered entry with some spread, and folds each generation's successes into
the memory weighted by how much they improved. Values that have been working recently are drawn
more often, and the learned values move with the search. It has been the basis of the strongest
differential evolution entrants in optimization competitions for a decade.

## The decision

`de_adapt_mutation = 1` turns the adaptation on for `de` and `ade`; `de_adapt_memory` (default
6, L-SHADE's) is the memory size. Off by default, and off, the family makes exactly the random
draws it made before, so an existing configuration runs as it did.

The pieces, and the two places the published method had to be adapted to PyBNF:

* **The memory starts at the configured pair.** Every entry begins at `mutation_rate` /
  `mutation_factor`, so the keys keep a meaning: they are where the learning starts, and a run
  in which nothing ever succeeds keeps drawing around them. SHADE's own initial value of 0.5 for
  both is PyBNF's default pair.
* **The draw is SHADE's.** A slot at random; the rate normal around it with spread 0.1, clipped
  to [0, 1]; the factor Cauchy around it with the same spread, capped at 1, drawn again while
  not positive. The update is SHADE's too: the rate as the improvement-weighted mean of the
  successful rates, the factor as the improvement-weighted Lehmer mean, which leans toward the
  larger factors because a small step succeeds more often but by less and the arithmetic mean
  would shrink the step every generation. A generation with no success writes nothing.
* **The record travels with the candidate.** `ade` returns results in whatever order the
  simulations finish, so the pair that built a candidate, and what it is to be judged against,
  are stored keyed by the candidate itself (the same value-keyed lookup `de`'s `island_map`
  already relies on) and popped when its result arrives. `de` perturbs a candidate that
  duplicates one in flight; the record follows the perturbed candidate.
* **A generation is what each method already calls one.** `de` folds an island's successes in
  when that island's generation ends; `ade` when a population's worth of results has arrived,
  the boundary at which it already prints its progress. `de` keeps one memory per island, since
  the islands are meant to search independently; `ade` keeps one. A multi-start run starts each
  start's memory afresh, because the settings a finishing search ends on are not the ones a
  beginning search wants.
* **A success is judged against the candidate's base, not the slot it competes for.** This is
  the adaptation that matters. SHADE crosses the mutant with the parameter set the candidate
  will replace, so the set a success is judged against is also the set the settings were
  applied to. PyBNF crosses the mutant with the base p1, which under the `rand` and `best`
  strategies is a different member from the slot the candidate competes for. Judged against the
  slot, a candidate built at a rate near zero is a copy of p1 and beats the slot whenever p1 is
  the better member, about half the time, by the whole gap between two members, without its
  settings having done anything. The history would learn that a rate near zero is best, and the
  population would thin out into copies of its best members. Judged against the base, a copy is
  never a success, and a success means what SHADE means by it: the settings improved on the set
  they were applied to. Under the `all` strategies the base is the slot and the two readings
  coincide.
* **One parameter is always mutated.** Binomial crossover guarantees this and PyBNF's crossover
  did not. With a learned rate that can sit near zero, the guarantee is what keeps a candidate
  from being an exact copy of its base, which could never be a real success: under the default
  seed policy it ties its base exactly, and under any other it beats it only by noise. It
  applies only when the settings are learned, so the off path is untouched.
* **Not adopted: the linear population-size reduction of L-SHADE (Tanabe and Fukunaga 2014).**
  It is a separate idea from the settings adaptation, and it interacts with how many processors
  a run keeps busy, which for `ade` is the point of the method. It should be judged on its own.

## The evidence

Two analytical targets, run through the test harness with no simulator, five seeds each, a
population of 30 for 150 generations (4,500 evaluations), the `rand1` strategy and the default
pair 0.5 / 0.5 as the fixed settings and as the memory's start. "Slot" is the alternative
reading of success, against the population slot the candidate competes for. Median best
objective over the seeds; the objective's minimum is 0.

| target | method | fixed settings | learned, judged against the base | learned, judged against the slot |
|---|---|---:|---:|---:|
| Gaussian, 10 parameters, independent | `de` | 0.135 | 0.000056 | 0.00018 |
| Gaussian, 10 parameters, independent | `ade` | 0.163 | 0.00018 | 0.00018 |
| Rosenbrock valley, 10 parameters | `de` | 8.3 | 3.8 | 4.2 |
| Rosenbrock valley, 10 parameters | `ade` | 4.5 | 3.9 | 4.5 |

Learned against the base, every seed on every target beat the fixed settings' median, and on
the valley the spread across seeds narrowed from 3.8 to 19.3 (`ade`) to 3.4 to 4.2. The learned
rate settled between 0.3 and 0.5 and the factor between 0.75 and 0.9 on both targets. Judged
against the slot, the learned rate drifted to between 0.10 and 0.25 on most seeds and the final
population held as few as 23 distinct members of 30 on the valley and 24 on the Gaussian, where
the base reading kept 29 or 30 on every seed: the cloning the design predicts, already visible
at 150 generations and worse at the objective on three of the four rows.

On the stochastic recovery benchmark (ADR-0140): six problems, five fit seeds each, the
frozen budgets, `de` and `ade` with and without the learned settings from the same seeds
(`results/de_adapt_5seeds.json`; the `de` rows are the baseline's, whose code path this
change leaves byte-identical). Success is every identifiable parameter within a factor of two
of the truth; the tight success is within 26 percent. "Only it" counts the seeds where one
variant succeeded and its pair did not; the error column counts seeds where the learned run's
final error was lower or higher than the fixed run's. The p values are two-sided sign tests.

| method | successes of 30 | tight successes of 30 | only it / only the fixed run | final error better / worse | median final error (decades) |
|---|---:|---:|---|---|---:|
| `de` | 6 | 0 | | | 0.630 |
| `de_adapt` | 11 | 5 | 6 / 1 (p = 0.13) | 17 / 13 (p = 0.59) | 0.573 |
| `de_forced` (control, below) | 8 | 4 | 6 / 4 (p = 0.75) | 16 / 14 (p = 0.86) | 0.540 |
| `ade` | 4 | 1 | | | 0.706 |
| `ade_adapt` | 8 | 4 | 5 / 1 (p = 0.22) | 15 / 15 (p = 1) | 0.833 |
| `ade_forced` (control, below) | 10 | 5 | 8 / 2 (p = 0.11) | 18 / 12 (p = 0.36) | 0.527 |

Where the gain is: Hlavacek_PNAS2001, four parameters and every one with high leverage on the
objective, goes from two successes of five to five of five under both methods, with final
errors of 0.006 to 0.10 decades against 0.14 to 1.5, and nine of those ten learned fits are
within 26 percent of the truth where no fixed fit was. McKane_PhysRevLett2005 gains two
successes under `de`. Where it is not: Yang_PhysRevE2008, the network-free problem with three
parameters and a budget of 4,000 simulations, is worse on four of five seeds under both
methods, without a success either way; Shahrezaei_PNAS2008 under `de` keeps its four successes
but is worse on four of five seeds. On the gene-expression problems whose parameters the data
barely see (Lin_PhysRevE2016, Munsky_Science2012) nothing changes that five seeds can resolve.
The gain is therefore in successes, and where the objective has a clear signal; the final
error pooled over every seed is not significantly different either way.

Plain `ade` stopped on its own before the budget in 24 of its 30 fits, after 3,300 to 19,200
simulations, having collapsed its population to a single parameter set. A candidate built at
rate 0.5 from a four-parameter model is an exact copy of its base one time in sixteen; under
the default seed policy the same parameters draw the same trajectories, so a copy ties its
base exactly and replaces any worse slot; and once every member is the same set, the
convergence test sees a spread of zero. `de` never did this within the budget, and with the
learned settings, which never propose an exact copy, `ade` did it in 4 of 30 (a copy can still
arise through a difference vector that is zero in the forced parameter, once members share
coordinates). This is a weakness of `ade` on small models independent of this change, worth
its own issue, and it means the `ade` comparison above measures the copy guarantee as well as
the learned settings.

To separate the two, a control ran each method with the settings fixed at 0.5 / 0.5 and only
the guarantee added (`de_forced` and `ade_forced` in the results file: not PyBNF fit types but
a subclass whose `new_individual` is the base's with the forced parameter and no history, run
under the same seeds and budgets). The guarantee alone never stopped early, and on its own it
raised the successes from 6 to 8 under `de` and from 4 to 10 under `ade`. Against that control,
the learned settings add successes under `de` (8 to 11; seeds only the learned run won 7,
only the control 4; final error better on 17 of 30) and take them away under `ade` (10 to 8;
3 against 5; final error worse on 19 of 30, p = 0.20). Under both methods the learned
settings' gain is Hlavacek_PNAS2001, five of five against three (`de`) and two (`ade`) for
the control, with final errors five to ten times smaller; elsewhere they are level with the
control under `de` and behind it under `ade`, most clearly on Shahrezaei_PNAS2008 (four
control successes against two) and McKane_PhysRevLett2005.

So on these six stochastic problems at five seeds: the learned settings help where the
objective has a clear signal and the run has room to learn, and cost a little where the
noise dominates or the budget is short; and the larger part of what the whole feature does
for `ade` is that it never proposes an exact copy, which is not a property of the learning at
all. That second finding belongs to `ade` and `de` regardless of this change, and is left for
its own issue: the guarantee could be made unconditional under the modern edition, as
ADR-0137 did for the diverse reference set, with the numbers above as its evidence.

## Consequences

* `DEFamilyConfig` gains `de_adapt_mutation` and `de_adapt_memory`; both `de` and `ade` own
  them. `DifferentialEvolutionBase` draws, records and folds; each subclass supplies its
  fitness list and says when a generation ends. The effective-config golden follows.
* `new_individual` takes an `island` argument so `de` can name the memory a candidate draws
  from. Its two existing arguments and its off-path behaviour are unchanged.
* At the end of the run PyBNF reports the memory's mean pair next to the pair the run started
  from, and at verbosity 2 prints it with each iteration's population summary, so a user can
  see what the run settled on and carry it into a fixed-setting run if they prefer one.
* The benchmark's `de` rows stay as measured; the learned-setting rows are in
  `results/de_adapt_5seeds.json`, alongside `ade` with and without it, which the baseline had
  not scored. Wall times in that file are not comparable with the baseline's: these runs
  shared the machine with each other and with the test suite.
* The key stays off by default, as the issue asked. The benchmark's case for it is a gain in
  successes at five seeds that a sign test does not yet confirm, part of which the copy
  guarantee alone accounts for, alongside a problem that got worse; the analytical case is
  unambiguous. Ten or twenty seeds on the four cheap problems, against the guarantee-only
  control rather than against the fixed settings, would settle the default.
* The copy guarantee on its own (the control) is the stronger change for `ade` on this
  benchmark and fixes its early stops; it is not shipped separately here, since the issue's
  scope is the learned settings and their off path is byte-identical. Filed as follow-up work.

## Alternatives considered

* **Judge success against the slot, as the issue's wording suggested.** Measured above: it
  learns to clone. Rejected on the mechanism and on the numbers.
* **Cross the mutant with the slot when learning, making the candidate SHADE's exactly.** That
  changes the search operator as well as the settings, so the two effects could not be told
  apart, and a user turning the key on would get a different algorithm, not the same one with
  learned settings. Judging against the base keeps PyBNF's operator and changes only what the
  history is told.
* **One shared memory across `de`'s islands.** Simpler, and more successes per update; but the
  islands are meant to search independently, the updates would land at every island's
  generation end and shorten the memory's reach in generations by the island count, and a
  memory per island is what a distributed SHADE would keep.
* **Update the memory one success at a time for `ade`, as the issue suggested.** That would
  store single successes rather than a generation's weighted means, losing the Lehmer bias that
  keeps the factor from shrinking. Folding at the boundary `ade` already marks keeps the
  published update.
* **Adopt L-SHADE's population-size reduction with it.** See above; separate idea, judged
  separately.
* **On by default.** The issue asked for off, so an existing configuration reproduces its
  behaviour exactly, and the benchmark's evidence is at five seeds. The default can follow once
  the evidence is wider.
