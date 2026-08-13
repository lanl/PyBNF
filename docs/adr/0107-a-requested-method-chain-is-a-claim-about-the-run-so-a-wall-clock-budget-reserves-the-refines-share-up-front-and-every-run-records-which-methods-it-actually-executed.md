# A requested method chain is a claim about the run, so a wall-clock budget reserves the refine's share up front and every run records which methods it actually executed (issue #564)

**Status: Accepted and implemented (2026-08-12).** `refine = 1` requests a *method* — search
globally, then polish locally — and under `wall_time_fit` it was silently downgraded to the
global search alone, essentially always. A new global key `wall_time_refine_frac` (default
`0.1`) holds that share of the budget back from the search, so the polish runs on a slice the
search was never allowed to spend; one deadline still bounds the whole run. Independently of
the mechanism, every run now writes `Results/method_chain.json` — the requested chain, the
executed chain, and each phase's status, stop reason, simulations, and best objective — and a
refined run's `sorted_params_final.txt` finally describes the refined point.

## The problem

A wall-clock-budgeted search runs until the clock stops. It has no reason to leave anything
behind, so the post-fit refine — new work, forbidden once the budget is spent (ADR-0093) —
never started:

```
Wall-time budget reached: stopped after 0:25:00 (wall_time_fit = 1500 s) and 33239 completed
simulation(s), with a best objective of -167.0476499.
Wall-time budget spent, so the best fit was not refined.
```

That was **15 of 15 runs** in a benchmark campaign (`Borghans_BiophysChem1997`,
wshlavacek/BNGL-Models#38) configured as `cmaes` + `refine = 1, refine_method = gntr`. Every one
of them actually ran plain `cmaes`.

Two contracts were in direct conflict and the budget won silently:

* ADR-0093: no new work starts once the budget is spent, and the run's outputs are complete.
* The conf: run this method chain.

### The downgrade was undetectable from the outputs

ADR-0093's central promise is that a budgeted run "writes exactly what a converged one writes",
so `sorted_params_final.txt` and `information_criteria.txt` look identical either way. The only
signal was a `print1` line on stdout. A harness that scores a directory — ours reads
`information_criteria.txt` — could not tell whether the method it believed it had measured had
run. For a benchmark corpus that is a provenance defect: we would have published "solved by a
cmaes+gntr hybrid" for a result produced by cmaes alone.

`Results/stop_reason.txt` did not close the gap. It says the *run* stopped early; it does not
say which phases that cost, and a run whose search finished on its own criterion and whose
*refine* was then cut off wrote no stop reason worth reading at all — the refiner's
`_announce_stop_reason` opened the shared file in `'w'` mode and overwrote the fit's.

### And a refine that did run could be a fraction of a polish

`refiner.budget = alg.budget` gave the refine whatever the search happened to leave. Four
seconds left meant a four-second polish, indistinguishable in every artifact from a converged
one.

## The decision

### The reserve is taken from the budget, not added to it

The issue laid out five options. Two are ruled out by the contract they break: a separate
`wall_time_refine` key makes the run's total `wall_time_fit + wall_time_refine`, and
"guarantee M refiner iterations regardless of the clock" overruns the deadline outright. Both
cost the one property `wall_time_fit` exists to provide — that the run fits the allocation.
Refusing the combination at config time is honest but leaves the user with no way to have what
they asked for.

So the search phase gets `(1 - f) * wall_time_fit` and the refine gets the rest, `f =
wall_time_refine_frac`, default `0.1`. The run's total is unchanged, the phase split is stated
in the conf, and ADR-0093's "one deadline for the whole run" holds exactly.

The rejected fifth option — sizing the reserve from measured evaluation cost — is the one most
likely to be *right* and the one least likely to be *predictable*. A reserve a user cannot read
off their conf reintroduces, in a subtler form, the problem being fixed: not knowing what the
run will actually do. A fraction is arbitrary; it is also legible, and a user who finds a tenth
too little can say so in one line.

### The reserve lives on the budget object, as a phase-relative deadline

`FitBudget` gains a `reserve`, and `remaining()` / `expired()` become the *current phase's*
deadline rather than the run's. Nothing else in the run loop changed: the search asks the same
`expired()` it always asked, and stops short of `wall_time_fit` because the object it asks now
knows a later phase has a claim on the tail.

`spend_reserve(budget)` is a context manager, not a one-way `release()`, for a reason that only
shows up with bootstrapping: replicates alternate fit and refine, so a reserve consumed by the
first replicate's polish would leave every later replicate's polish unprotected. Releasing for
the duration of a phase and restoring on the way out gives every replicate the same split. It
is a no-op on `None`, so an unbudgeted run needs no guard and takes no new branch.

The whole of `_refine_best_fit` — including its "is the budget spent?" test — runs inside that
block. Asking the *search's* deadline whether the refine may start is precisely the bug: the
reserve would be held back and then never spent.

### The reserve is a floor under the refine, not a cap on it

Inside the block the budget is the run's whole remaining time — the reserve plus whatever a
search that converged early left behind. A search that hits its own stop criterion at 30% of
the budget hands the other 70% to the polish.

The converse is real and accepted: a refine that converges in a tenth of its reserve ends the
run with the rest unspent. Under a fixed allocation that is wasted compute — but the method the
conf requested has completed, and the alternative (returning to the global search after the
polish) is a *different* method chain, not the one that was asked for. The reserve is bounded
by `f`, so the waste is bounded by `f`.

### A reserve is taken only when a refine will actually be attempted

No `refine`, no budget, or a `refine_method` naming the algorithm the fit itself ran — which
PyBNF has always skipped — all leave the search the entire budget. A fit that asks for no
polish is byte-identical to pre-#564, and the reserve never funds a phase that could not run.

### The remaining skip is loud, and names the key that would prevent it

The skip is still reachable: `wall_time_refine_frac = 0` (the explicit opt-out), or a search
that overran its share because one in-flight simulation outlived the deadline by up to
`wall_time_sim` (ADR-0093 documents that overrun). It is now `print0` — every verbosity level —
it states that the run executed the global search *alone*, and it points at
`wall_time_refine_frac`. A silent downgrade to a different method is the defect; an explicit
one is a choice.

### Every run records the method chain it executed

The mechanism above fixes the reported case. It does not make any run *self-describing*, which
is the property the benchmark harness actually needs, so `Results/method_chain.json` is written
by every run — budget or no budget, downgrade or no downgrade:

```json
{
  "format_version": 1,
  "job_type": "cmaes",
  "wall_time_fit": 1500,
  "refine_reserve_seconds": 150.0,
  "requested_methods": ["cmaes", "gntr"],
  "executed_methods": ["cmaes"],
  "phases": [
    {"phase": "fit", "method": "cmaes", "status": "wall_time_expired",
     "reason": "Wall-time budget reached: ...", "elapsed_seconds": 1350.4,
     "simulations": 33239, "best_objective": -167.0476499, "bootstrap_replicate": null},
    {"phase": "refine", "method": "gntr", "status": "skipped",
     "reason": "the search overran the 150 s reserved for the refine", ...}
  ]
}
```

Four decisions inside that file:

* **A new file, beside the results — not a new field inside them.** This is ADR-0093's own
  resolution of "indistinguishable to tooling" versus "not mistaken for convergence", applied
  again: no existing format changes, so no existing parser learns anything.
* **Written for every run, not only for the ones that went wrong.** Provenance a consumer can
  only rely on when something failed is provenance it cannot assert on. `requested_methods` vs
  `executed_methods` is a one-line assertion in a scoring harness.
* **Rewritten after every phase.** A run whose refine raises still leaves the record of the fit
  that did happen.
* **JSON, and strictly valid.** A non-finite objective (`inf`, from a run whose every
  simulation failed) is recorded as `null` rather than `Infinity`, which a strict parser would
  reject — losing the whole record over one field.

Bootstrap replicates are recorded, but with a `bootstrap_replicate` index that keeps them out
of `executed_methods`: "which methods did this run execute" is a question about the run, not
about replicate 17. A final `bootstrap` phase carries `replicates_requested` /
`replicates_completed`, because `bootstrap = 30` in a conf is worth nothing if the budget
stopped the run at 11.

The record is filled in by `pybnf.pybnf`, not by `Algorithm.run`. A run's phases are an
orchestration concept, and the orchestrator is the only place that sees all of them.

### Two adjacent provenance defects, fixed here because they are the same defect

* **`sorted_params_final.txt` described the pre-refine point.** The refiner wrote its result to
  `sorted_params_refine_final.txt` while rewriting `information_criteria.txt` from the same
  end-of-run tail — so two files in one `Results/` disagreed about which parameter set they
  described, and the *conventional* name carried the point the requested method chain did not
  end on. A refine's end-of-run output is the run's end-of-run output; it is now written under
  both names. The `refine_`-prefixed file is unchanged, so anything already reading it keeps
  working.
* **A bootstrap replicate's refine wrote into the main run's `Results/`.** `_refine_best_fit`
  already redirected the replicate's `sim_dir` and `failed_logs_dir` to `Results-boot{N}`'s
  peers but not its `res_dir`, so every replicate's polish overwrote the *main* fit's
  `sorted_params_refine_final.txt` — and, with the alias above, would have overwritten
  `sorted_params_final.txt` too. The refiner now writes where the fit it is polishing wrote.

And `_announce_stop_reason` appends for a refine rather than truncating: a run where the search
hit the deadline and the polish did too has two facts to report, not one that replaces the
other.

## Consequences

* **`wall_time_fit` + `refine = 1` executes the chain it names.** The benchmark campaign that
  motivated the issue can be re-run and the results labelled honestly.
* **A budgeted run's search is ~10% shorter by default.** That is the cost of the polish, and
  it is stated on the console before the search starts as well as in the stop reason. A user
  who wants the old split writes `wall_time_refine_frac = 0` and is told, loudly, what it
  costs them.
* **Nothing changes for a run that names no budget.** No `FitBudget`, no reserve, no new
  branch — the ADR-0093 default path is untouched.
* **`sorted_params_final.txt` changes meaning for refined runs**: it is now the run's final
  point, not the search's. It agrees with `information_criteria.txt`, which it did not before.
  A consumer that wanted the pre-refine trajectory can read the fit's periodic dumps.
* **Every run gains one small file.** `Results/method_chain.json` is a few hundred bytes for an
  ordinary run; a `bootstrap = 1000` run records 2000 phases and is correspondingly larger.
* **The reserve can still be wasted or overrun.** A polish that converges early leaves its
  remainder unspent, and one in-flight simulation can eat into the reserve (ADR-0093's
  documented overrun). Both are bounded, and both are now visible in the record rather than
  silent.
* **A per-phase deadline was *not* introduced.** `wall_time_fit` remains the run's only
  deadline; the reserve partitions it rather than adding to it.

## Verification

* `tests/test_wall_time_fit.py` — the reserve on the stopwatch (a phase-relative
  `expired()`/`remaining()`, a search that converges early handing its leftovers on,
  `spend_reserve` restoring the reserve even when the phase raises, a reserve clamped to the
  limit); the sizing rule across all six ways it can come out zero or non-zero; the config
  surface (a number in `[0, 1)`, with `1` refused alongside the negatives); and the **headline
  regression** — a `de` fit whose ticking clock would consume the whole budget now stops at
  `wall_time_fit - reserve`, says so in its stop reason, and is followed by a refine that
  actually runs with time to spend, with the split intact afterwards for the next replicate.
  Plus its inverse: `wall_time_refine_frac = 0` reproduces the pre-#564 downgrade exactly.
* `tests/test_method_chain.py` — the record object (written as each phase lands, skipped phases
  excluded from `executed_methods`, bootstrap replicates kept out of the run's chain, a
  non-finite objective serialized as `null`, a write failure that does not take the run down,
  the requested chain read off a conf including the `refine_method == fit_type` case); the
  artifacts (a real DE fit followed by a real Simplex refine, asserting
  `sorted_params_final.txt` carries the *refined* best objective and matches
  `sorted_params_refine_final.txt`, and that the refine's stop reason is appended to the fit's
  rather than replacing it); and end to end, that a completed chain records
  `["de", "sim"]` while a downgraded one records `["de"]` against a request for both.
* `tests/golden_configs/effective_config_golden.json` regenerated and
  `test_benchmark_harness._EXCLUDE` extended for the new global default — the standard "a
  global key was added with a no-op default" updates.

## References

* Issue #564 — the report: the 15-of-15 campaign, the five options, and the two asks that were
  to hold "independently of the mechanism".
* ADR-0093 / issue #529 — `wall_time_fit`, the one-deadline-per-run contract this partitions,
  the "writes exactly what a converged one writes" promise that made the downgrade invisible,
  and the documented overruns.
* ADR-0015 / issue #403 — `refine_method`: the refine as a first-class choice of local
  optimizer, which is what makes `refine = 1` a method request rather than a flag.
* ADR-0056 — `information_criteria.txt`, the artifact a scoring harness reads and the one that
  `sorted_params_final.txt` now agrees with.
* ADR-0091 — a refusal states its reason *and* its remedy; the shape of the skip warning.
* Grein et al. 2026, bioRxiv 2026.07.11.737731 — the wall-time-budgeted benchmark protocol the
  campaign was run under.
