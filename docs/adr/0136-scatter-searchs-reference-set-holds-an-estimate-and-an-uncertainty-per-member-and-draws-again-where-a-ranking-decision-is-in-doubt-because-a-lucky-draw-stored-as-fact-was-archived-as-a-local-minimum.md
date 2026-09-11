# Scatter search's reference set holds an estimate and an uncertainty per member and draws again where a ranking decision is in doubt, because it only ever needs the ordering to be right and a lucky draw stored as fact was being archived as a local minimum (issue #660, step 3)

## Status

Accepted and implemented (2026-09-10). Step 3 of the #660 epic, taken first because it is
the one step that fixes a fault in shipped behaviour, and because the piece it needs
arrived with ADR-0135. Steps 1, 2 and 4 of the epic are separate changes. Amended by
ADR-0141 (2026-09-11): the `ss_noise_max_draws` default is 3, not 5, on the stochastic
recovery benchmark's measurement of the deferral as the feature's whole cost.

## The fault

Scatter search never needs its objective values to be precise. Every decision it makes is
a ranking decision: whether a child beat its parent, whether a member has been stuck for
`local_min_limit` rounds, and the rank gap between two members, which sets the step size
of their combination through `beta = (|hi - pi| - 1) / (popsize - 2)`.

For a stochastic model each value is one draw, and the reference set stored that draw as
fact. A member whose value was a lucky draw could not be beaten by honest children, whose
single draws are typically worse than its lucky one, so its stuck counter climbed until
`local_min_limit` and it was retired into `local_mins`, the archive of local minima, as one
it never was. That archive is what the run prints as its best archived scores, so on a
stochastic model it filled with artifacts of the noise, and the member's slot was refilled
from the reserve rather than improved.

## The decision

### An estimate and an uncertainty per member, with the noise pooled across the fit

Every reference member keeps its draws and is ranked on their mean. The draw-to-draw
spread of the objective is one number for the fit, the pooled within-parameter-set
standard deviation over everything drawn more than once
(`pybnf.algorithms.noise_handling.pooled_sd`), on the assumption that the spread is about
the same everywhere a search visits. That is what gives a single draw of a new child an
uncertainty at all. Two estimates are *separated* when their means differ by more than one
standard error of the difference, `sd * sqrt(1/n_a + 1/n_b)` (`separated`); a non-finite
mean is separated from anything, an unknown spread separates nothing, and a zero spread
separates everything that differs at all, which is the deterministic case.

The pooled spread is bootstrapped by drawing every member of the first reference set once
more, alongside the first round of children, so the first decisions already have it.

### A decision in doubt is not made

For each parent, the best candidate is the best of its new children (single draws) and the
contender left over from a previous round in doubt (with its draws). A candidate that is
better than the parent and separated from it replaces it, and the candidate's draws travel
with it. One that is worse and separated counts the parent stuck. One that is not separated
leaves the contest open: it becomes the parent's contender, both sides are drawn again, and
the decision waits for the next round. When both sides have spent `ss_noise_max_draws`
draws the means decide, so a contest cannot stay open forever.

A parent counted stuck is drawn again as well, up to the cap. A member that keeps beating
its children is exactly the one whose recorded value is being trusted, and this is what
makes a lucky value regress to its true value, so that an honest child can beat it, or so
that, if it really is a local minimum, it is archived at its true value.

After the reference set is sorted on means, each adjacent pair the spread cannot order is
drawn again too, since its rank gap sets a step size. All re-draws of a round go out with
the round's children and are bounded by the cap, so a member costs at most
`ss_noise_max_draws` simulations over its life beyond the ones the search would have spent
anyway.

### A fresh draw is a fresh trajectory

A re-draw is a copy of the parameter set named for the draw it is, carrying a replicate
offset past every draw the set has had (ADR-0135's `PSet.replicate_offset`, honoured by
`make_job`), so under the default seed policy it is a new trajectory rather than the same
one again, and its smoothing group cannot collide with an earlier draw's.

### Where this does nothing

The handling turns on only when `_replicates_would_differ()`, the gate ADR-0108's
confirmation stage and ADR-0135 use, and `ss_noise_handling` is on (the default). A
deterministic fit keeps every draw list at length one, never queues a re-draw, and makes
every decision by the same strict comparison as before: it is byte-identical, including
the names and order of the parameter sets it queues. The reference set's public shape, a
list of `(pset, score)` pairs, is unchanged, so the white-box tests that place members by
hand still hold; a member placed without draws is estimated from the score it carries.

What the trajectory records is unchanged: every draw is its own entry, and the end-of-fit
confirmation stage settles the reported answer (#659).

## What this is not

Step 4 of the epic, giving an idle processor another draw of a member whose rank is still
uncertain instead of waiting for the slowest simulation, is the natural next use of the
same bookkeeping and is not built here: this round's re-draws go out with the round. Steps
1 and 2 (the improvement method and choosing the diverse half by distance) are independent
of noise and separate changes.

## Consequences

* `ScatterSearchConfig` gains `ss_noise_handling` (default 1) and `ss_noise_max_draws`
  (default 5 here; 3 since ADR-0141), registered in the parse layer, the docs and the
  effective-config golden.
* `ScatterSearch` gains `draws`, `contenders` and `pending_draws`; `_search_got_result`
  routes a re-draw to its parameter set and defers the round while any re-draw is in
  flight; `_update_reference_set`, `_count_stuck`, `_unseparated_neighbours` and
  `_redraws` carry the decisions.
* `pybnf.algorithms.noise_handling` gains `pooled_sd` and `separated`, beside the CMA-ES
  measurement, which is the shared component #661 asked for.

## Verification

`tests/test_scatter_noise.py`: the pooled spread and the separation test on hand-built
values; a deterministic fit queues no re-draw and decides as before; a stochastic fit
bootstraps the spread with one re-draw of every member at a fresh offset; the fault itself,
a member with a lucky first draw and honest children, archived at its lucky value with the
handling off and drawn again, regressed and beaten with it on; a contest in doubt kept open
and settled by re-draws; the cap ending a contest on the means; a stuck member drawn again
up to the cap; unordered neighbours drawn again, ordered ones not, and none twice in a
round; the state pickling and resetting between starts; and the whole run loop through the
integration harness with a noisy fake runner, which still finds the mode.

## Prior art

Egea et al. (2009) and Penas et al. (2017) for the scatter search template PyBNF follows,
neither of which treats a noisy objective. Hansen, Niederberger, Guzzella and Koumoutsakos
(2009), IEEE Transactions on Evolutionary Computation 13(1), 180-197, for the principle that
a ranking-driven search should measure its ranking's reliability and spend evaluations only
where it is low (ADR-0135); this ADR applies the same principle to a reference set through
per-member estimates rather than a population-level rank-change statistic.
