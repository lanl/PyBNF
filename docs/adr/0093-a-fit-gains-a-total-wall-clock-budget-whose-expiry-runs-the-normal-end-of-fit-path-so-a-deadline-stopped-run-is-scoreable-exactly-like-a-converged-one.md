# A fit gains a total wall-clock budget whose expiry runs the normal end-of-fit path, so a deadline-stopped run is scoreable exactly like a converged one (issue #529)

**Status: Accepted and implemented (2026-08-02).** A new global config key `wall_time_fit`
(seconds; `0` = unbounded, the default) bounds the **whole run**, the peer of the two existing
per-unit-of-work limits `wall_time_sim` and `wall_time_gen`. When it expires the run loop stops
launching work, abandons what is in flight, and then runs the **same** end-of-fit path a converged
run takes — `sorted_params_final.txt`, the best-fit simulations, `information_criteria.txt`, the
ArviZ sidecar, the backup rename, the sim-dir teardown — against the best point found so far. The
only difference from a converged run is the stop *reason*, which is logged, printed, and written to
`Results/stop_reason.txt`. A fit that names no budget is byte-identical to pre-#529.

## The problem

PyBNF's only time limits were per unit of work:

```python
# pybnf/parse.py
'wall_time_sim',   # one simulation
'wall_time_gen',   # one network generation
```

Neither bounds a *run*. The only native run-level budget was `max_iterations` × `population_size`,
which is not convertible to wall time without knowing per-iteration cost in advance — and that cost
varies by orders of magnitude across problems, and within a problem as the search moves through
stiff regions.

### 1. The published optimizer benchmark allocates compute by wall time

Grein et al. 2026 (bioRxiv 2026.07.11.737731; 33 optimizers × 30 PEtab problems, >1.5M core-hours)
run "simple problems on 12 cores with a wall-time limit of 3 hours, challenging problems on 24 cores
with a wall-time limit of 9 hours", launching multi-start runs until the limit. The whole
leaderboard is *best objective reached within budget B*; first-hitting-time and overall-efficiency
comparisons are defined against that budget. Without a wall-time budget PyBNF could not be run under
that protocol at all — a matched comparison was unavailable, not merely untuned.

### 2. An externally killed run cannot be scored

Absent a native budget, the only way to stop at a deadline is to kill the process — which loses the
result, because the artifacts scoring needs are written only on normal completion. Scoring needs the
full normalized log-likelihood behind `OG = -log_likelihood - J*` ("solved" iff `OG < 1.92`), and
`-log_likelihood` comes from `information_criteria.txt`, emitted only at the end of a successful
fit:

| situation | `sorted_params_*.txt` | `information_criteria.txt` | scoreable |
|---|---|---|---|
| normal completion | yes (`_final`) | yes | yes |
| uncaught exception | yes (`_end`, cleanup path) | **no** | no |
| SIGTERM / SIGKILL | only the periodic dumps, if any | **no** | no |

The periodic dumps carry the best-so-far *reduced* objective, which is not convertible to `-lnL` for
any problem with an estimated noise scale (the dropped constant is parameter-dependent there). So
the number that survives a kill is genuinely not the number needed. Measured on the 18 unsolved
subset-I problems under an external 300 s cap: **13 produced no scoreable result**, two of them
without even a first periodic dump.

The finalize half is therefore the load-bearing half. A deadline that leaves an unscoreable
directory is barely better than the external `kill` it replaces.

## The decision

### The budget is an object with a deadline, not a config value read at each site

`pybnf/budget.py` holds a `FitBudget` — a monotonic stopwatch with a limit — and `main()` builds one
(or `None`) once, from `wall_time_fit`, and attaches it to the algorithm. Three consequences follow
from making it an object rather than a config read:

* **Unbounded is `None`, not a sentinel limit.** Every consumer asks `budget is not None`, so the
  default path has no arithmetic in it at all and cannot drift from the historical behavior.
* **One deadline spans the phases.** The same object is handed to the refiner and reused across
  bootstrap replicates, so `wall_time_fit` bounds the *run*, not each phase separately. A per-phase
  reading would let a `refine`-and-bootstrap job spend an arbitrary multiple of its stated budget.
* **The clock is `time.monotonic`, seeded with the process's own start time.** A system-clock
  adjustment mid-fit cannot lengthen or shorten a budget; and because the wall-clock origin `main()`
  already records is folded in as an `elapsed` offset, configuration loading and network generation
  are *inside* the budget — the budget bounds what an external `timeout` around the process would
  have bounded, which is what makes it comparable to the benchmark's allocation.

### Expiry runs the *existing* tail; there is no separate "budget finalize" path

This is the decision the issue actually turns on. Expiry `break`s out of the run loop at the same
place the algorithm's own `'STOP'` does, so `run()`'s tail is reached identically and unchanged.
Nothing is conditional on *why* the loop ended. A budgeted run is not "a partial run with some
outputs" — it is a finished run whose search was cut short, and every downstream consumer
(`information_criteria.txt`, the best-fit BNGL artifact, the ArviZ bridge, a benchmark scorer) sees
what it always sees. Anything less than reusing the tail verbatim would have re-created the very
gap the issue reports, one artifact at a time.

### Three enforcement points, because "stop at the deadline" and "launch nothing new" are different claims

1. **After each completed result, before submitting its successors.** The result in hand is recorded
   first — it is already paid for — and then the budget decides whether anything new may launch.
2. **As `as_completed`'s `timeout`.** The loop blocks in `next(pool)`; with only check (1) an expiry
   would not be noticed until some simulation happened to finish, which on a stiff problem could be
   an hour. `distributed.as_completed` already takes a `timeout` and raises `TimeoutError` out of
   `__next__` once it elapses, so the remaining budget is handed to it at construction (the deadline
   is absolute and `update()` does not reset it — exactly the semantics wanted). The in-flight
   simulations are then abandoned via the `client.cancel` the loop already does at teardown. The
   kwarg is passed **only** when a budget exists, so an unbudgeted fit's call is the historical one.
   (`timeout`, and the `TimeoutError` raised out of `__next__` while the queue is empty, are present
   unchanged at our declared `distributed>=2024.1.0` floor, so this is not a version-conditional
   path.)
3. **Before the first submission.** Setup can eat the whole budget on a model whose network
   generation is slow. An expired budget means *launch nothing*, so the initial generation is not
   submitted either; the run goes straight to the tail. (This is what motivated extracting the drain
   loop into `_drain_job_pool`, which also makes the three exits testable as one unit.)

Two things are deliberately **outside** the budget, and are documented rather than engineered away:
one in-flight simulation may overrun the deadline by up to `wall_time_sim` before its abandonment
takes effect, and finalizing re-simulates the best fit once (that re-simulation is how
`information_criteria.txt` gets the full normalized log-likelihood). Both are small and bounded next
to a multi-hour allocation, and the alternative — refusing to finalize once the clock is out — would
defeat the purpose.

### The stop reason is durable, and lives outside the scoreable artifacts

"Indistinguishable to downstream tooling" and "not silently mistaken for a converged run" pull in
opposite directions. They are resolved by putting the reason *beside* the results rather than in
them: `Results/stop_reason.txt`, written **only** when a run stops for a reason that is not its own
stop criterion. Its mere presence is the signal a harness keys on; no existing file's format changes,
so no parser has to learn anything. The message is in universal terms — elapsed time, completed
simulations, best objective — because there is no iteration counter shared by every fit type (a
generation, a start, and a chain step are all "one iteration" to different algorithms), and an
approximate "N iterations" would be a worse claim than an exact "N completed simulations".

### A fit type that cannot honor the budget refuses it

`wall_time_fit` is honored by every fit that drives the shared run loop. `hmc` does not — it runs
blackjax NUTS in process over an analytical target, with no per-simulation dispatch point to stop at
— so naming the key there is a configuration error with a hint pointing at that job's own budget keys
(`num_warmup` / `num_samples` / `num_parallel`), not a silent no-op (the ADR-0091 rule: a refusal
states its reason *and* a remedy). `check` is different in kind — one evaluation of given
parameters, not a search — so the key is stripped there alongside `refine` and `bootstrap`, which
`check` already drops.

### A resumed run gets a fresh budget

The budget is excluded from the backup pickle: a wall-clock deadline restored into a later process
is meaningless, and a `--resume` is a new run of its own. `main()` builds the resumed algorithm a
fresh budget from the same `wall_time_fit`, so `-r` grants another full allocation rather than
inheriting an already-spent one. (Exclusion is via `should_pickle`, and `budget` is a **class**
attribute defaulting to `None`, so an unpickled algorithm reads a well-defined value.)

### A run with no results reports that, instead of raising

`Trajectory.best_fit()` is `max` over a heap and is undefined on an empty trajectory, so a run that
stopped before its first result — newly reachable, when a budget expires during the first
generation — would have raised `ValueError`/`IndexError` out of an otherwise-finished run. The tail
now asks `len(self.trajectory)` (a new `Trajectory.__len__`) and says "no simulation completed, so
there is no best fit to report", writing no parameter file rather than an empty, unloadable one.
This also covers the pre-existing path where the job pool is exhausted before any result.

## Consequences

* **PyBNF can be run under the Grein et al. protocol.** `parallel_count = 12` + `wall_time_fit =
  10800` is the paper's "12 cores × 3 h" allocation, and the run it produces is scoreable by the
  same rule as every other optimizer on that leaderboard.
* **A budgeted run and a converged run are indistinguishable to tooling.** Same files, same formats,
  same one-extra-simulation provenance for the likelihood. The difference is one additional file.
* **A deadline is now a supported way to end a fit**, so an external `kill` — which loses the
  result — is no longer the only option for a fit that is not converging.
* **No new behavior for anyone who does not ask for it.** With `wall_time_fit` unset there is no
  budget object, no `timeout` on `as_completed`, no extra check that can fire, and no new file.
* **The budget is not a hard process deadline.** It bounds the *search*; the overruns above (one
  in-flight simulation, one finalize re-simulation) are real, and a harness that also imposes an
  external cap should leave headroom for them.
* **`refine` and further bootstrap replicates are new work**, so a spent budget skips them. A
  budgeted fit can therefore report an unpolished best fit, or fewer bootstrap replicates than
  requested — both are stated on the console, and the replicates already accepted are complete on
  disk (`bootstrapped_parameter_sets.txt` is appended as each finishes).
* **`hmc` refuses the key** rather than accepting a deadline it would never honor.
* **A method's *own* end-of-run step, if it lives in its `'STOP'` branch, does not run.**
  `run()`'s shared tail is what a budget stop guarantees; a per-method wrap-up inside `got_result`
  (`am`'s `combine_chains_params` writing `Results/A_MCMC/Runs/combined_params.txt`, for instance) is
  reached only by that method's own stop criterion. The per-chain draws and the periodic diagnostics
  are on disk either way, so nothing is lost that cannot be recomputed, and a budgeted sampler run
  still gets `samples.txt` and the shared tail. Hoisting those steps into an overridable
  `finalize_search()` hook is the obvious generalization, deliberately not taken here: it would mean
  auditing every sampler's `'STOP'` branch for double-execution, which is a wider change than the
  budget, and no consumer needs it yet (the ≥2-user bar of ADR-0009).

## Verification

* `tests/test_wall_time_fit.py` — the stopwatch on an injected clock (expiry exactly at the limit,
  the pre-existing `elapsed` charge, `from_config`'s `None` for unbounded); the config surface (a
  non-negative integer required, the `hmc` refusal carrying both diagnosis and remedy, the `check`
  strip); an **end-to-end** budgeted `de` fit of a 2-D Gaussian through the real `Configuration` and
  the real algorithm, given a `max_iterations` it could never reach, which is ended by the deadline
  alone and still writes `sorted_params_final.txt` and `stop_reason.txt`; and the post-fit phases —
  a spent budget starts no refine and no further bootstrap replicate, a live one is handed to the
  refiner as the *same object*.
* `tests/test_run_loop.py` — the loop's three exits under the fake dask client: expiry while a
  result is in hand (that result is still recorded; nothing new is submitted; the tail still runs),
  the `TimeoutError` path with every worker busy (no results at all — abandoned, finalized, and
  reported as having no best fit rather than raising), a budget already spent before the first
  submission (not one job submitted), the recorded stop reason and its file, the remaining budget
  reaching `as_completed` as its `timeout`, and the unbudgeted control: no timeout, no stop reason,
  no file, runs to its own `STOP`.
* `tests/golden_configs/effective_config_golden.json` regenerated for the new global default
  (`wall_time_fit: 0`), and `test_benchmark_harness._EXCLUDE` extended for the same reason — both
  are the standard "a global key was added with a no-op default" updates.

## References

* Issue #529 — the report: the benchmark protocol, the artifact table above, and the 13/18
  measurement.
* Grein et al. 2026, bioRxiv 2026.07.11.737731 — the wall-time-budgeted benchmark protocol.
* ADR-0007 — the run-loop contract (`start_run` / `got_result`, one shared loop) the budget stops
  inside, rather than asking each method to honor a deadline of its own.
* ADR-0091 — a refusal states its reason *and* its remedy; the shape of the `hmc` refusal.
* ADR-0056 — `information_criteria.txt`'s full normalized log-likelihood, the number a budgeted run
  must still emit.
