# Scatter search gives a processor that would idle at the end of a round another draw of a member whose rank is in doubt, because the idle time and the noise share one answer (issue #660, step 4)

## Status

Accepted and implemented (2026-09-11). The last step of the #660 epic, after steps 3
(ADR-0136), 2 (ADR-0137) and 1 (ADR-0138).

## The problem

Scatter search runs `popsize * (popsize - 1)` simulations per round and waits for every one
of them before it builds the next round (`waits_for_full_generation`). Toward the end of a
round only the slowest few are still running, so processors sit idle. For a stochastic
model this is worst of all: a simulation's running time depends on the trajectory it happens
to take, so the spread in running times is wide and the idle tail is long, and it is exactly
the stochastic model whose ranking needs more draws.

## The decision

### The run records its processor count

`Algorithm.run` reads how many simulations the run can execute at once when it starts,
summing the threads of every worker dask reports, for a local cluster and a cluster run
alike, and keeps it as `worker_count`. The parallelism report read the same number but did
not keep it, and skipped local runs. `None` when it cannot be read; nothing depends on it
being known.

### Idle processors get the draws the round would have spent anyway

When a result comes back mid-round and fewer jobs are in flight than the run has
processors, the difference is filled with fresh draws of the members whose rank the noise
cannot settle: the neighbours in the sorted reference set that ADR-0136 would draw again at
the round's end, plus both sides of every open contest. They are the same draws, taken
earlier, on processors that had nothing else to do, so the round's decisions are sharper
when it ends and no simulation was spent that the noise handling would not have spent.
Every draw stays within `ss_noise_max_draws`, so a member costs no more over its life
however the draws are timed, and a draw already queued this round is not queued twice.

The round still waits for those draws, as it waits for its children. A draw issued at the
very end of a round can extend it by about one simulation, which is the price of using the
processor at all; a draw is short next to the round it fills.

### Where this does nothing

A deterministic fit has no noise handling and fills nothing. `ss_fill_idle = 0` fills
nothing while leaving the rest of the noise handling on, for a fit that would rather end
each round as soon as its own simulations do. A run whose processor count could not be read
fills nothing. A round with every processor busy fills nothing. A refinement
in flight (ADR-0138) counts as work in flight, so its jobs are never crowded out.

## Consequences

* `Algorithm` gains `worker_count` and `_count_workers`, set once at run start.
* `ScatterSearchConfig` gains `ss_fill_idle` (default 1), registered in the parse layer,
  the docs and the effective-config golden.
* `ScatterSearch._search_got_result` returns `_fill_idle_processors()` mid-round instead
  of nothing.
* `waits_for_full_generation` stays true: the round still synchronizes; what changes is
  what the idle processors do while it does.

## Verification

`tests/test_scatter_idle.py`: the processor count sums every worker's threads and is unknown
when unreadable or empty; a deterministic fit never fills; nothing without a count or while
every processor is busy; idle processors get draws of the members the noise cannot order, at
the next replicate offset, registered as pending; the cap holds and the round still completes;
and both sides of an open contest are drawn. The existing scatter search, noise-handling,
diversity, improvement-method, run-loop and integration tests pass unchanged.
