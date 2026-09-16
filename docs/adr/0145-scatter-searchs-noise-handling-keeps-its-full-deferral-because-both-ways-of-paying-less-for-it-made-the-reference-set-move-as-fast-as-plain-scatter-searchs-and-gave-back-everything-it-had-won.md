# Scatter search's noise handling keeps its full deferral and its unrationed re-draws, because both ways of paying less for them made the reference set move as fast as plain scatter search's and gave back everything the deferral had won (issue #696; completes ADR-0141)

## Status

Accepted and implemented (2026-09-16), both keys off by default under every edition. Closes
the two levers ADR-0141 left open. The plan was that one of them would beat the current
default and become it; the measurements below did not support that, and both ship as opt-in
keys with the numbers recorded, which is the other outcome #696 allowed for.

## The problem

ADR-0136 gave scatter search a noise-aware reference set: a parent-versus-child decision the
noise leaves in doubt is not made, both sides are drawn again, and the contest waits. The
stochastic recovery benchmark then measured the deferral as the whole cost of the feature
(ADR-0141), which is why its default fell from five draws to three. The reasoning was that
while a contest is open the old parent goes on seeding the next round's combinations, so the
reference set evolves about half as fast as plain scatter search's, and that the re-draws come
out of the same simulation budget as the search. Two ways of paying less were named and left
untried:

* **Accept a candidate that leads on the mean while the draws continue.** Plain scatter search
  would put the better-looking child in the set at once. The correction the noise handling
  exists for can still happen, because the contest goes on and the draws can hand the slot
  back.
* **Draw again only for the contests that matter.** Every open contest and every unseparated
  neighbour pair is drawn again each round, and they are not worth the same: a contest between
  two estimates a hundredth apart costs almost nothing to call wrongly, and one at the top of
  the reference set costs more than one at the bottom.

Both were built, both work as designed, and both made the fits worse.

## What was built

### `ss_noise_optimistic`: the leading side holds the slot while the contest runs

A contest in doubt whose candidate leads on the mean puts the candidate in the reference set
at once and makes the member it displaced its contender. Both sides keep being drawn, within
`ss_noise_max_draws` as before, and the contest is decided as it always was: the displaced
member is a candidate for the slot it used to hold, so if the draws settle the other way it
takes the slot back. A candidate that does *not* lead changes nothing.

The slot's stuck counter restarts when the candidate takes the slot, exactly as it would for
a settled replacement, because the slot's point changed. Carrying the count across the
takeover was tried first and is wrong: the round in which the contest settles in the new
occupant's favour is recorded as a round in which nothing replaced the slot -- the winner
already holds it -- so the count would climb on a contest *won*, and a member would be retired
into the archive of local minima after a few wins. That is a large change to
`local_min_limit`'s meaning and nothing to do with the lever.

### `ss_noise_redraw_budget`: a round keeps open only the contests worth keeping open

The round is now planned before it is applied. Every slot's decision is worked out first
(`_plan_slot`, which writes nothing), the contests the noise leaves in doubt are ranked, and
only the highest-ranked ones the budget reaches are kept open and drawn for. The rest are
decided now on their means, exactly as a contest at the cap is decided -- *not* left open
undrawn, which would lengthen the deferral rather than pay less for it.

The rank is what a wrong call would cost: the objective gap between the two estimates,
weighted by where in the sorted reference set the decision sits, linearly from the best member
to the worst. The gap is the whole of the cost, because it is how much worse the reference set
is if the wrong side wins; how *likely* that is separates these pairs far less, since each one
is within a standard error of the other, which puts the chance of a wrong call between about
one in six and one in two whatever the gap. The weight was not tuned.

Two re-draws stay outside the budget. The draw of a member counted stuck is the one that keeps
a lucky value out of the archive of local minima, the fault ADR-0136 was written to fix, and
there is at most one per slot per round. The draws a processor takes rather than idling
(ADR-0139) cost the round no work it could have done instead. A round's own decisions are
served before its neighbour pairs, because a contest decides which point seeds the next round
while a neighbour pair only sets a step size.

## The evidence

Both levers were scored on the same four problems and the same twenty seeds as ADR-0141, at
the deferral of 3 that is now the default, paired against the `ss_noise_d3` rows already in
`results/ss_noise_20seeds.json`. Pairing is sound because a fit is deterministic in
(problem, method, seed): a fresh `ss_noise` fit of McKane_PhysRevLett2005 from seed 1 under
this change reproduces the committed record to the last digit, which is also the check that
the restructuring above left the default path alone.

### The mechanism does what it was built to do

One McKane_PhysRevLett2005 fit (seed 1, 20 rounds, 2,000 evaluations), instrumented:

| setting | re-draws | slot changes | open contests | rounds |
|---|---:|---:|---:|---:|
| `ss_noise` (deferral 3) | 161 | 69 | 95 | 20 |
| `ss_noise_optimistic = 1` | 192 | 113 | 94 | 19 |
| `ss_noise_redraw_budget = 4` | 127 | 97 | 46 | 20 |
| `ss_noise_redraw_budget = 2` | 99 | -- | 25 | 20 |

"Slot changes" is how many times a reference slot's point changed over the fit, which is the
quantity ADR-0141 named as the cost: it found plain scatter search accepting 91 and 105
replacements per run where the noise handling at a deferral of five accepted 60 and 45. At a
deferral of three it is 69. Optimistic acceptance raises it to 113 and a budget of 4 to 97,
both inside plain scatter search's range, and the budget does it while spending a fifth fewer
re-draws. Whatever else follows, neither lever failed to move the number it was aimed at.

### The fits got worse

Eighty paired seeds, against the current default:

| method | successes of 80 | median final error | mean final error | error better / worse | sign p | only it / only `ss_noise_d3` | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ss_noise_d3` (the default) | 24 | 0.674 | 0.682 | | | | |
| `ss_noise_opt` | 17 | 0.676 | 0.729 | 39 / 40 | 1.000 | 7 / 14 | 0.189 |
| `ss_noise_b4` (budget 4) | 15 | 0.597 | 0.708 | 37 / 43 | 0.576 | 3 / 12 | 0.035 |
| `ss_noise_b2` (budget 2) | 13 | 0.706 | 0.738 | 39 / 41 | 0.911 | 2 / 13 | 0.007 |

The budget is significantly worse than the default on success at both settings it was tried
at. Optimistic acceptance is not significantly worse on either statistic, but it is worse on
both, and its 17 successes against 24 have no compensating gain anywhere in the tables.

Against plain scatter search, the same eighty seeds:

| method | successes of 80 | error better / worse | sign p | only it / only `ss` | McNemar p |
|---|---:|---:|---:|---:|---:|
| `ss` | 17 | | | | |
| `ss_noise_opt` | 17 | 33 / 47 | 0.146 | 7 / 7 | 1.000 |
| `ss_noise_b4` | 15 | 36 / 44 | 0.434 | 7 / 9 | 0.804 |
| `ss_noise_b2` | 13 | 35 / 45 | 0.314 | 5 / 9 | 0.424 |

This is the sharpest way to put it. A deferral of three is the one setting that beat plain
scatter search on successes, 24 against 17. Under either lever the noise handling is
indistinguishable from not having it. Both levers give back exactly what the deferral won.

### Faster early, worse at the end

The error of the reported best at checkpoints through the budget, median over the twenty
seeds, shows what the faster-moving reference set buys and what it costs. On
Shahrezaei_PNAS2008, the problem whose noise misleads a single draw most and where the default
wins:

| method | 2000 | 5000 | 10000 | 15000 | 20000 |
|---|---:|---:|---:|---:|---:|
| `ss_noise_d3` | 0.600 | 0.466 | 0.333 | 0.263 | **0.250** |
| `ss_noise_opt` | 0.592 | 0.409 | 0.306 | 0.295 | 0.367 |
| `ss_noise_b4` | 0.548 | 0.363 | 0.289 | 0.338 | 0.383 |
| `ss_noise_b2` | 0.437 | 0.362 | 0.342 | 0.347 | 0.323 |

Both levers lead at every early checkpoint and lose at the last two. Lin_PhysRevE2016 does the
same, both ahead at 2,000 and 5,000 and behind at 20,000. A reference set that moves faster
does find a better answer sooner, and then settles on a worse one.

## What this says about the deferral

ADR-0141 read the slower-moving reference set as the feature's cost, to be recovered if a way
could be found. Two ways were found, both recovered it, and the benefit went with it. So the
reading was wrong: the slow movement is not overhead beside the better ranking, it *is* the
better ranking. Holding a slot while its contest is undecided is what stops the set from
following a lucky draw, and drawing again for contests that look not worth deciding is what
stops it from following several at once. The checkpoint curves say the same thing another way:
spend the noise handling's caution and the search gets ahead early on estimates it has not
earned, and cannot keep the lead.

That leaves the deferral length as the only lever on this feature that has ever paid, which is
where ADR-0141 left it.

## Consequences

* `ScatterSearchConfig` gains `ss_noise_optimistic` (default 0) and `ss_noise_redraw_budget`
  (default 0), registered in the parse layer, both documentation files, the key-ownership
  tests and the effective-config golden. A configuration that does not set them runs exactly
  as before.
* `_update_reference_set` is plan-then-apply: `_plan_slot` decides a slot without writing,
  `_contests_to_keep_open` ranks the contests in doubt, and the apply loop runs in slot order
  so the archive and the reserve are consumed in the order they always were. With both keys
  off, every decision, every re-draw, and the order they are queued in are unchanged, verified
  against a committed benchmark record.
* `run_baseline.py compare` is the comparison the #660 and #663 studies did by hand, now in
  the runner. It reproduces their published tables from the committed records.
* The 240 new fit records are in `results/ss_noise_20seeds.json` beside the others, and the
  tables are in the benchmark README.

## Alternatives considered

* **Make one of them the default.** The plan. Nothing in the tables supports it: one is
  significantly worse on success, the other worse on every statistic without significance.
* **Delete both rather than ship them off by default.** #696 asked for named toggles so the
  variants could be scored by name, and they are how the next person re-runs this on more
  problems or a longer budget without rebuilding them. ADR-0144 shipped a measured-negative
  key on the same reasoning. The cost is two integers in a schema and one planning pass that
  is a no-op when they are unset.
* **Optimistic acceptance with a longer deferral.** If the trouble is that the set moves too
  fast, a longer cap might pay for the faster movement. It might, but ADR-0141 already measured
  five as the worst deferral tried, so this asks for a two-dimensional scan on evidence that
  neither dimension helps on its own. Not run.
* **Rank the contests by expected regret rather than by the gap.** The gap times the chance of
  a wrong call, with the chance from the separation statistic. Among the pairs that reach the
  ranking the chance varies by a factor of three and the gap by far more, so it would reorder
  few decisions, and the budget's problem is not its ordering: a budget of 4 and one of 2
  differ by a factor of two in what they ration and both lose the same way.
* **Ration the stuck member's re-draw too.** It is 6 percent of the re-draws in the
  instrumented fit and the one that fixes ADR-0136's fault. Rationing it would remove almost
  nothing and risk the thing the feature exists for.

## Verification

`tests/test_scatter_noise.py` adds two classes. For optimistic acceptance: it is off by
default and inert on a deterministic fit; a leading candidate takes the slot with the
displaced member as its contender and both drawn again; the slot's stuck count restarts; a
slot that wins its open contest is not archived for winning it; the draws settling the other
way hand the slot back with the candidate's draws dropped; a candidate that does not lead
leaves the parent in place; and a settled contest is decided as it always was. For the budget:
it is off by default; the priority is the gap weighted by rank, clamped at the last slot; with
no budget every request is queued in the order the decisions were made; a budget spends on the
highest priorities and stops; a request the draw cap has emptied costs nothing; the stuck
re-draw is never rationed; a round keeps open only the contest it can afford and accepts the
rest on their means, with the neighbour pairs finding the budget spent; and a whole fit with a
budget draws fewer times than one without.

`tests/test_stochastic_recovery_benchmark.py` adds the comparison: the exact two-sided
binomial, the sign test dropping ties including two failures, McNemar on the discordant seeds
only, pairing restricted to the seeds both methods ran, the reported best at a checkpoint, and
`compare` end to end with its default checkpoints, an explicit set, and a problem subset.

## Prior art

Egea et al. (2009) and Penas et al. (2017) for the scatter search template PyBNF follows,
neither of which treats a noisy objective. Hansen, Niederberger, Guzzella and Koumoutsakos
(2009) for spending evaluations where a ranking is least reliable, which is what the budget's
ranking is a cheaper form of; the result here is that scatter search's reference set wants the
caution spread over every decision in doubt rather than concentrated on the expensive ones.
