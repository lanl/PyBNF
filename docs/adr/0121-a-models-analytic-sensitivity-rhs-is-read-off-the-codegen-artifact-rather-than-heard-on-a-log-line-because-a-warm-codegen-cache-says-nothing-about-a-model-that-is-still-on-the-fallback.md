# A model's analytic sensitivity RHS is read off the codegen artifact rather than heard on a log line, because a warm codegen cache says nothing about a model that is still on the fallback (issue #606)

**Status: Accepted and implemented (2026-08-20).** Answers the question ADR-0119's
consequences section deferred: where a worker-side `∂f/∂p` decline reaches the user.

`CVodeSensInit1` takes **one** sensitivity-RHS callback for every column, so a single rate
law bngsim cannot differentiate declines the analytic `∂f/∂p` for the *whole* model — there
is no per-reaction fallback to mix in. CVODES' internal difference quotient then carries
every column. That substitution preserves correctness and multiplies cost: one extra RHS
evaluation per column per step, so an N-column request pays roughly N times the sensitivity
cost.

On a fit measured in hours the cost is what ends runs. On `Smith_BMCSystBiol2013` all 25
columns fell back, every start timed out to `inf`, and thirteen hours produced nothing. The
only signal was a bngsim log line nobody had a reason to look for.

## The gap

bngsim says so, on the `bngsim` logger, at codegen time:

> Forward sensitivity: *&lt;reason&gt;*, so the analytic sensitivity RHS is declined for this
> model and CVODES' internal difference quotient is used instead (correct, but slower).

PyBNF converted that into nothing a user of a fit would see. No console line, no entry in
the run's own reporting, no refusal, and nothing distinguishing a gradient fit on the
analytic path from one on the fallback.

The line was not *lost*. `bngsim`'s logger propagates to root, PyBNF's `init_logging` puts a
`FileHandler` there, and `Cluster` runs `init_logging` on every worker, so it lands in
`<prefix>.log`. But that is a shared, noisy file written from N worker processes, and the
line arrives once per model at first codegen, mid-run. It is discoverable by someone who
already suspects the problem, which is the wrong order.

## The finding that decided the design

Issue #606's first suggested direction was to capture the `bngsim` logger during the
gradient path and re-emit a decline as a PyBNF-level warning. It needs no new bngsim API and
works across the whole supported range, which is exactly what recommends it.

**It does not work, and it fails in the silent direction.** Since lanl/bngsim#174 the codegen
cache key is *structural*, so a warm cache resolves the `.so` from a handful of C++ reads and
**generates no source at all** — and source generation is where the decline is derived and
logged. Measured with an `abs()` rate law, in one process, on **bngsim 0.14.0** — the current
release, and the one CI resolves — and reproduced identically on 0.13.0:

| construction | verdict (artifact) | decline logged |
|---|---|---|
| first, cold cache | fallback | yes, with the reason |
| second, warm cache | fallback | **nothing** |

That the two builds agree is worth more than either measurement alone, because they do not
agree by taking the same path: 0.14.0 answers through route 2 below and 0.13.0 through route 3,
so one measurement exercises both rungs of the ladder and shows they return the same verdict.

The cache lives on disk and persists across runs, so the construction that hears nothing is
typically the *second run of the same fit* — the run a user makes after the first came back
empty. A design that took its verdict from the log line would report that run as fine.

It is not even bngsim's cache alone. PyBNF's own `_ENGINE_TEMPLATE_CACHE` holds one engine
model per SBML text per process, and a template that has been through a sensitivity
construction carries its `_codegen_so_path` — so the *second model in a worker* skips codegen
for the same reason, with no disk cache involved at all. Two independent caches, one shared
consequence, which is what makes this a property of the channel rather than a bug in either.

So silence on that logger means "declined, or served from cache". It can support a statement
that a model **is** on the fallback, and never one that it is not.

## The decision

**Read the verdict off the codegen artifact; keep the log line for the reason.** The two
channels answer different halves of the report, and the split is the whole ADR:

| | channel | reliability | may be acted on |
|---|---|---|---|
| **verdict** | does the artifact export `bngsim_codegen_sens_rhs`? | exact, cache-hit or not | yes — policy keys off it |
| **reason** | bngsim's own decline line, captured during construction | present on a cold cache only | prose only |

`bngsim_codegen_sens_rhs` is the exact symbol bngsim's C++ resolves with `try_symbol` to
choose the analytic RHS over the difference quotient, so reading it back off the artifact
answers the question for the run that is actually about to happen. `analytic_sens_rhs_probe`
resolves it through four routes, first match wins, and reports which one answered — the same
ladder, and the same "prefer what the build publishes" rule, as ADR-0119's
`_resolve_event_sens`:

| # | route | when it answers |
|---|---|---|
| 1 | `Simulator.has_analytic_sens_rhs` | if bngsim ever publishes a per-run answer — **both** directions, ahead of everything else |
| 2 | `Simulator._codegen_provides_sens_rhs()` | bngsim's own ground truth; private, ≥ 0.14.0 |
| 3 | the symbol in `_codegen_c_source` / the `.so` at `_codegen_so_path` | today, on every build in the pin |
| 4 | no artifact to read | **no opinion** |

Route 1 fires on no build that exists. Naming it costs nothing and means PyBNF starts reading
the real answer on the first build that grows one, with no PyBNF release in between; that is
issue #606's "related upstream", lanl/bngsim#431. Route 2 is preferred over route 3 wherever
it exists because it is upstream's answer to upstream's question — if the symbol ever stops
being where route 3 looks, a build carrying the method keeps answering correctly. Route 3 is
the load-bearing one: the floor admits 0.13.0, where route 2 does not exist, and a feature
that only works on the newest build would not have caught the failure that motivated it.

**Route 4 is a real answer and is reported as one.** A `codegen=False` run integrates the
interpreted RHS and has no artifact; an unreadable `.so` is a different unknown. Guessing is
wrong in both directions — a false *present* hides the cost this whole probe exists to
surface, and a false *absent* warns about a fit that is fine — so no opinion is logged and
never warned about. That is a departure from ADR-0119's fail-closed rule, and deliberately:
there the wrong guess buys a converged wrong answer, here it buys a wrong sentence.

### Where it runs: the head node, at gradient-path setup

Issue #606 named the placement problem precisely — the fact is decided at codegen, which
happens on a **worker** at the first sensitivity-bearing `simulate()`, not on the head node
at `_setup_gradient_path()`. The natural reading is that the answer has to be found on a
worker and carried back on a result.

It does not. Reading the artifact needs a Simulator, and `_setup_gradient_path()` can build
one: the models are constructed, the sensitivity request is applied, and it is the head node.
So `_report_sensitivity_rhs()` builds **one sensitivity-bearing Simulator per model** there
and probes it. That buys three things a worker-side capture could not:

* the answer arrives **before the fit has spent anything**, which is the whole complaint the
  log line could not answer — thirteen hours is not the cost of learning this too late, it is
  the cost of learning it at all;
* no worker→head plumbing, no per-model de-duplication across N processes, and no decision
  about whether a decline rides back on a `Result` or is logged in place;
* a `sensitivity_fallback = error` policy becomes possible, because there is a point in the
  run at which refusing is still free.

The cost is one codegen per model at job start. It is close to free where it matters: the
codegen cache is content-addressed and shared, so on a local run every worker then gets a
cache hit and the compile has been *moved* rather than added. On a cluster with no shared
cache directory it is one extra compile per model, against a fit measured in hours — the same
trade `_report_bngsim_build()` already makes ("a fit is hours; the warning is worth one line
at job start").

`probe_sens_rhs` **never raises**. A model that cannot be prepared for a sensitivity run
reports no opinion, because a diagnostic that can end a fit is worse than no diagnostic;
whatever is genuinely wrong surfaces at the first simulation with its own message.

### The policy knob

`sensitivity_fallback` takes `warn` (default), `error`, or `ignore`.

`warn` names the model, its column count, the cost multiplier, and bngsim's reason when it
gave one — to the log, and to the console, which is the point. `error` refuses instead, for a
long unattended run. `ignore` skips the check entirely, including the one Simulator
construction it costs, which is the reason to have an `ignore` at all.

**The policy keys off the verdict, never off the reason.** That is not a detail. The verdict
is stable across a cold or warm cache; the reason is not, so a policy that keyed off it would
refuse a fit on its first run and accept the same fit on its second. Reproducible refusal is
worth more than a more informative one.

`error` also does not refuse a model that reported **no opinion**. The knob refuses a *known*
fallback; it cannot promise to detect one it could not read, and refusing an unreadable build
would take down a `codegen=False` run that is perfectly fine.

### The half where "correct, but slower" is false

There is a decline whose difference quotient does not answer the same question: the model
branches at a crossing whose time moves, and the fallback integrates the variational equation
straight through it, dropping the saltation jump `(f⁻−f⁺)·dt*/dθ`. Every column is then wrong
at and after that crossing.

From 0.14.0 bngsim **refuses** such a run — `SensitivityUnsupportedError`,
lanl/bngsim#414/#416 — rather than return a gradient it has flagged as wrong, so there it
reaches PyBNF as an ordinary error and needs nothing new. On 0.13.0, which the floor still
admits, the same model only warns and returns the wrong gradient.

Worth noting *how* 0.14.0 decides it, because it is this ADR's argument arrived at
independently: `Simulator._raise_if_uncompensated_crossing_sensitivities` opens with `if
self._codegen_provides_sens_rhs(): return` and then re-scans the model for the crossing. It
does not consult the codegen warning at all — so its refusal survives a warm cache, and the
comment beside it reaches the same conclusion, that the artifact is the ground truth and a
re-derivation is not.

PyBNF reports the case as a statement about the **numbers** rather than the cost: a `print0`,
past the verbosity the cost line respects, saying the columns are wrong at and after the
crossing. The wording has to hold on both sides of the 0.14.0 line — on a carrying build the
fit is about to stop at the first simulation, on an older one it is about to run — so it says
bngsim refuses the case from 0.14.0 and asks for a finite-difference check *if this fit
proceeds*.

It is **not** made a PyBNF refusal, and the reason is the cache again. Being carried on the
reason channel, it can only fire when the reason was heard — so a refusal here would fire on a
fit's first run and not on its second. A non-reproducible refusal is worse than a consistent
warning. This is the same rule as the policy knob, applied to the case where it costs the most
to obey.

## What this does not claim

* **It is not a statement about the whole fit.** One verdict in bngsim's codegen pipeline
  reads parameter values and species initials (the switch gate), so a model branching on a
  fitted threshold can in principle cross the line mid-search. The probe describes the model
  at the fit's start point. Probing every evaluation is not a trade worth making; the caveat
  is written down instead.
* **It is not a per-condition statement.** A `condition:` perturbs parameter *values*, not
  the rate-law expressions that decide differentiability, so the wildtype probe stands for
  the model — subject to the same start-point caveat.
* **It does not make PyBNF choose the path.** PyBNF cannot supply an analytic `∂f/∂p` bngsim
  declined; what it can do is stop the user paying for the discovery in wall-clock.

## Consequences

* The rule this adds to the one stated in ADR-0119 — *a capability is decided by what the
  build says it can do; a version string is at most a veto* — is about **channels**: a
  diagnostic channel that can be short-circuited by a cache may carry explanation but not
  verdict. The asymmetry has to be checked before a channel is trusted, and here checking it
  reversed the issue's own first suggestion.
* Route 3 replicates a private upstream method for the builds that lack it, and says so.
  Route 2 exists precisely so that replication retires itself on every build that carries the
  real thing, and route 1 retires both the day bngsim publishes a per-run key.
* The probe is a general capability read, so it lives in `_bngsim_caps` beside the others —
  keeping "one module asks bngsim what it can do" true even for a fact that is about a
  (build, model) pair rather than a build.
* The standing ask upstream is unchanged and now has a second consumer: a supported,
  non-private read of "is this model on the analytic path" (lanl/bngsim#431) would collapse
  routes 2 and 3 into route 1.
* Not addressed here: reporting the *measured* cost. The probe says a model is on the
  fallback and what the column count implies; it does not time a sensitivity solve, so
  "roughly Nx" stays an estimate from the difference quotient's own arithmetic rather than an
  observation.
