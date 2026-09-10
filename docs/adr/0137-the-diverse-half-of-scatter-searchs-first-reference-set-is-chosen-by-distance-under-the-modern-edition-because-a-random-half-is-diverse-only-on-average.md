# The diverse half of scatter search's first reference set is chosen by distance under the modern edition, because a random half is diverse only on average and Glover's template says to choose for it (issue #660, step 2)

## Status

Accepted and implemented (2026-09-10). Step 2 of the #660 epic, after step 3 (ADR-0136).
Steps 1 and 4 remain.

## The gap

Glover's scatter search template builds the first reference set from two halves: the best
of the initial population by quality, and the most *diverse* members of the rest, chosen by
their distance from what is already in the set. PyBNF's `round_1_init` took the best half
and then filled the rest with `rng.choice` over the remainder. A random half is diverse only
on average, and in a problem with many parameters that is much weaker than choosing for it:
the members that drive the combinations toward unexplored regions are exactly the ones a
random sample is unlikely to include.

## The decision

### Greedy max-min, in sampling space, per-coordinate normalized

The second half is filled one member at a time with the candidate farthest from the nearest
member already in the reference set, the standard greedy construction. Distance is Euclidean
in sampling space `u`, so a log-scaled parameter is measured on its log scale, and each
coordinate is divided by its spread over the whole initial population, so a parameter
declared over a thousand units does not drown one declared over one. Ties go to the better
score, since the candidates arrive sorted by it. A candidate whose score is not finite is
taken only when no finite one is left: a point the model could not simulate is not a useful
reference, however far away it sits.

### On under the modern edition, off under the legacy one

The rule changes which reference set a fit starts from, and so every parameter set it
queues afterwards. ADR-0031's contract is that a conf naming no edition keeps behaving as
it always has, so the rule is on under `edition = 2` and above and off under the legacy
edition, where the random choice stays byte-identical. `ss_diverse_by_distance` sets it
explicitly either way, which is also what makes the change measurable on its own, as the
issue asks: the same conf can be run with each rule.

## Consequences

* `ScatterSearchConfig` gains `ss_diverse_by_distance` (unset resolves by edition),
  registered in the parse layer, the docs and the effective-config golden.
* `ScatterSearch.round_1_init` branches on the rule; `_most_diverse` is the construction.
* The modern first round makes no random draw for its second half; the legacy one is
  unchanged.

## Verification

`tests/test_scatter_diversity.py`: each pick is the candidate farthest from the set so far,
ties go to the better score, coordinates are measured against their spread, a failed point
is taken last, the edition gate and the explicit key, the modern first round never draws
from the random generator while the legacy one still does, and a modern scatter search
still finds the Gaussian mode end to end.

## Prior art

Glover, F. (1998), A template for scatter search and path relinking; Egea et al. (2009)
and Penas et al. (2017), whose reference-set construction uses the same max-min diversity
rule.
