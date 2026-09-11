# Scatter search refines the best child a round accepts with a simplex run alongside the rounds, under Egea's filters, because Glover's template has an improvement method and ours had none (issue #660, step 1)

## Status

Accepted and implemented (2026-09-10). Step 1 of the #660 epic, after steps 3 (ADR-0136)
and 2 (ADR-0137). Step 4 remains.

## The gap

Glover's scatter search template has five methods: diversification generation, improvement,
reference set update, subset generation, and solution combination. PyBNF had four of them.
Egea and colleagues' enhanced scatter search, which PyBNF's combination formula comes from,
applies a local solver to promising combined solutions as well. Ours relied on recombination
alone, and the issue names this the largest gap and the most likely source of better fits.

The piece was already in the tree: `SimplexRunner`, the headless Nelder-Mead state machine
the concurrent multi-start drives, which works in parameter-set space, knows nothing about
the trajectory or the scheduler, and pickles.

## The decision

### What is refined, and when

The improvement method starts from the best child a round accepts into the reference set,
the point Egea's version refines. Three filters, Egea's, keep it from running on every
candidate:

* a **cadence**: not more than one start every `ss_local_every` rounds (default 10);
* a **concurrency cap**: not more than `ss_local_max_running` searches in flight (default
  1, the sequential search of the literature; more uses the processors a large cluster has
  idle at the end of a round);
* a **distance filter**: not from a point within 5 % of the initial population's spread per
  parameter (root-mean-square, in sampling space) of a refinement already started or an
  optimum already found, which is a basin already refined.

A child whose score is not finite is never refined.

### How it runs: alongside the rounds, never blocking one

A simplex is sequential: each step waits for the last. Making a round wait for one would
serialize the fit. So a refinement's jobs go out with the round's children, its results are
routed to it by a per-search tag (`ls<k>_`) whenever they arrive, and its next step is
queued at once; the round completes on its own children and re-draws only. On a cluster the
refinement therefore costs wall clock only when processors are short, which is what the
issue asks for.

The initial simplex's edge is a tenth of the reference set's spread per parameter, in
sampling space, falling back to a tenth of the initial population's, so a refinement starts
at the scale the search is working at rather than at the simplex fit type's fixed unit step.
The Nelder-Mead constants are that fit type's defaults, and a search ends at
`ss_local_max_iterations` simplex iterations (default 50) or when its largest move falls
below a small tolerance.

### How it comes back

At the next round boundary, each finished refinement's best point enters the reference set
in place of the member it started from when that member is still there and the point is
better; else in place of the worst member when it beats that; else it goes to the archive
of local minima, which is what a refinement that found nothing new is. Every optimum is
recorded for the distance filter, and the folded point is renamed for its search so that
re-draws under noise handling could never collide across searches.

### Off under noise handling

A Nelder-Mead simplex over single draws of a stochastic model converges on the noise rather
than the objective; a noise-aware local search is a different feature. The improvement
method is therefore off whenever noise handling (ADR-0136) is on, even when asked for, and
the log says so. On under `edition = 2` and off under the legacy edition otherwise, by
ADR-0031's contract, with `ss_local_search` setting it explicitly.

## Consequences

* `ScatterSearchConfig` gains `ss_local_search`, `ss_local_every`,
  `ss_local_max_iterations` and `ss_local_max_running`, registered in the parse layer, the
  docs and the effective-config golden.
* `ScatterSearch` gains the refinement bookkeeping and four methods:
  `_maybe_start_local_search`, `_advance_local_search`, `_fold_finished_local_searches` and
  `_replace_member`; `_update_reference_set` records the children a round accepts.
* A deterministic legacy fit is byte-identical; a modern one queues refinement jobs beside
  its rounds.

## What this is not

Step 4 of the epic, giving an idle processor another draw of a member whose rank is still
uncertain instead of waiting for the slowest simulation, remains. A noise-aware improvement
method is not proposed.

## Verification

`tests/test_scatter_local.py`: the gate (legacy off, modern on, the explicit key, off under
noise handling even when asked for), the knobs; nothing starts before a round accepts a
child; a refinement starts from the best accepted child with the initial simplex's vertices
tagged and the origin recorded; the initial simplex is a tenth of the reference set's
spread; the cadence, concurrency and distance filters; a failed child is not refined;
results route to the refinement and never end a round; a finished refinement replaces its
start, or the worst member, or is archived, according to its score; a straggler of a folded
refinement is dropped; the state pickles and resets; and a scatter search with the method
on finds the Gaussian mode end to end with refinement jobs among those it ran.

## Prior art

Glover (1998), A template for scatter search and path relinking, for the five methods; Egea
et al. (2009), Ind. Eng. Chem. Res. 48(9), 4388–4401, for the local-search filters; Lee and
Wiswall (2007) for the parallel Nelder-Mead PyBNF's simplex implements.
