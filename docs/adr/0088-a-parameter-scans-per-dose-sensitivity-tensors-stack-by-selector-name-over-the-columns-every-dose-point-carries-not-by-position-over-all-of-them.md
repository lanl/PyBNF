# A parameter scan's per-dose sensitivity tensors stack by selector name over the columns every dose point carries, not by position over all of them (issue #525)

**Status: Accepted and implemented (2026-07-30).** A dose-response (`type: parameter_scan`)
experiment runs each dose point as an independent simulation and `stack_scan_sensitivities`
(ADR-0046 / #476) stacks those per-point forward-sensitivity tensors down a dose axis so gradient
assembly can address the scan by dose row exactly as it addresses a time course by time row. It
stacked them **positionally** — `np.stack([p.d_param[-1] for p in per_point])` — with the first
point's selector list adopted as the label for all of them. That holds only while every dose point
reports the identical column set. When one point reported a different set, `numpy.stack` raised a
bare `ValueError: all input arrays must have the same shape` from inside PyBNF and the whole
gradient fit aborted. This ADR stacks **by selector name over the intersection**: each point's rows
are read through that point's own selector list, the stack covers the columns present at every
point, and a point that cannot be reconciled at all raises a `PybnfError` naming the dose index and
the shapes.

## The problem

The reporter's job (`Salazar-Cavazos-2019/egfr_simpull`, an EGF dose-response over three doses plus
a 25 nM time course, six `loguniform_var` parameters) died on `job_type = trf` inside 11 seconds,
before a single start completed:

```text
File "pybnf/bngsim_model/net_model.py", line 1192, in _run_parameter_scan
File "pybnf/data.py",                    line 86,  in stack_scan_sensitivities
File "numpy/_core/shape_base.py",        line 458, in stack
    raise ValueError('all input arrays must have the same shape')
```

The per-dose tensors were `(2, 34, 6)`, `(2, 36, 6)`, `(2, 36, 6)` — same times, same parameters,
**different column counts**. The model has 23 observables and 13 global functions; the 36-column
points carried all 13 functions and the 34-column point was missing two of them
(`random_pYpY_per`, `pYpYinPYN_per`). A different pair went missing on the next evaluation.

That set is not fixed per model. `_extract_output_sensitivities` asks the backend only for the
functions it says it can differentiate (`_differentiable_expression_names`, filtering bngsim's
per-`Result` `_expression_sens_support` verdict map), and bngsim decides that map **per
`Simulator`** — one per dose point, since each point is an independent reset-to-seed run. So the
column set is a per-point fact that PyBNF reads back per point, while the stacker treated it as a
per-scan constant.

The immediate trigger is an upstream defect: bngsim 0.11.35's `sympy_to_c` emits through a
process-wide cached printer whose resolver it assigns and then clears
(`_c_printer_cache[0]._resolver = resolve_symbol` … `finally: printer._resolver = None`), which is
not thread-safe. Concurrent emissions inside one worker process race on that attribute, and the
loser sees an unresolvable symbol and reports the function as non-differentiable. Measured directly
on the same arithmetic derivative that every one of these functions needs: **0 spurious refusals in
2000 serial calls, 14 in 4000 across 8 threads.** Every refusal seen in the reporter's job named an
ordinary quotient (`100*pY1068/EGFRtot` "derivative w.r.t. `EGFRtot` is not representable in C"),
and a function refused at one dose was accepted at the next — so these are false negatives, not
model properties.

Whether a worker runs concurrent threads at all is a PyBNF configuration accident: `cluster.py`
pins `threads_per_worker=1` when `parallel_count` is set, and otherwise takes dask's default
`Client()`, whose workers are multi-threaded. The reporter's config sets no `parallel_count`. Adding
`parallel_count = 4` to it — nothing else — takes the same `trf` fit from "aborts before the first
start" to five completed TRF iterations with no ragged point at all (zero column-drop warnings
across the run), which is both the confirmation of the mechanism and the workaround available
today.

Fixing that race belongs upstream and is tracked there. It is *not* what this ADR is about: the
stacker's positional assumption is a latent PyBNF defect that any per-point column difference
exposes, and it failed in the two ways a shared-shape assumption always fails.

**A ragged set killed the fit.** The abort was total — no start completed, no objective was
evaluated — and the message named `numpy.stack`, one layer below anything a user configures.
Worse, it was wrapped: `net_model.execute` blamed the *model*, "may use discrete events or
otherwise non-differentiable dynamics," which sent the reporter reading an event-free BNGL file
looking for a construct that was never there.

**A reordered set would have been silently wrong.** The same code takes `first.selectors` as the
label list for every point's rows. Two points carrying the same columns in a different order stack
without error and label dose 1's `expression:f` sensitivities `observable:A`. Nothing downstream can
detect that; the fit just walks a wrong gradient. Shape agreement was standing in for column
agreement, and it is not the same predicate.

## The decision

### Align by name, over the intersection

`stack_scan_sensitivities` computes the selectors present at *every* dose point, in the first
point's order, and reads each point's final row through that point's **own** selector list:

```python
selectors = _common_scan_selectors(per_point)          # order: point 0; membership: all points
cols = [point.selectors.index(s) for s in selectors]   # per point, not shared
rows.append(tensor[-1][cols, :])
```

The uniform case — every scan whose points agree, which is every scan when the backend is
behaving — produces the identical tensor it did before: the intersection is the full list, the
index map is the identity, and the stack is unchanged.

The intersection is the right cut because these columns are independent. A scan tensor missing
`expression:random_pYpY_per` is not an approximation of anything; it is a complete, correct tensor
over the other 35 columns. Dropping a column costs the fit **only if that column is scored**, and
then the gradient path already has the error it needs: `_selector_for` raises
`GradientNotSupported` — "No forward-sensitivity column for scored observable 'pY1068_percent'
(have: …)" — which names the observable, at the layer that knows what "scored" means. In the
reporter's model 3 of 13 functions are scored, so most raced-out columns now cost nothing at all.

The union was rejected outright: a column absent from a point is absent because that point's
backend declined it, and its slot in that point's tensor holds a NaN sentinel. Stacking the union
means stacking NaN and computing a NaN gradient — the one outcome worse than an abort.

### Say which dose point, and what it looked like

Every divergence is now reported at the layer that sees it, and the reports name the dose index:

* **Columns dropped** (recoverable) — a `WARNING` listing each absent selector and the dose points
  that lacked it: `expression:pY1173_percent absent at dose point(s) 2; expression:unphosR_per
  absent at dose point(s) 1`. That single line is the whole diagnosis of the reporter's bug, and it
  is what the issue asked for.
* **Nothing shared** / **an axis missing at a later point** — `None`, plus a warning. This reuses
  the established "cannot supply sensitivities at every dose point" contract: the scan carries no
  tensor and the gradient path reports the gap once, uniformly, rather than per point.
* **Irreconcilable** — a `PybnfError`, since no alignment can rescue it: disagreeing axis labels
  (`dose point 1 labels its d_param axis ['k1', 'k3'], but dose point 0 labels it ['k1', 'k2']`), a
  tensor whose extents contradict its own labels (actual shape *and* the expected one), or an empty
  row axis with no final row to contribute. These replace the numpy `ValueError` that the issue
  asked to stop surfacing raw.

A `PybnfError` also changes *how* the failure is presented, not just its text:
`net_model.execute` re-raises `PybnfError` as-is and wraps everything else. So a PyBNF-level
consistency failure now surfaces as itself instead of being dressed up as a backend/model problem.

### The model-construct diagnosis becomes conditional

The gradient-path wrapper in `net_model.execute` keeps its role — turn a backend raise into an
actionable PyBNF message — but leads with the underlying error and states the
non-differentiable-model reading as one possibility, alongside the failure report that names the
failing action and parameter set. It also points at `job_type`, the key a user actually sets.

## Consequences

* **A per-point column difference no longer aborts the fit.** Where the dropped column is unscored
  the scan is fitted normally; where it is scored the fit fails with the observable's name instead
  of a `numpy.stack` traceback. On the reporter's job the crash is gone and the warnings name the
  culprit columns and doses on every occurrence.
* **A reordered column set can no longer mis-label the tensor.** Positional stacking's silent
  failure mode is closed by construction, not by a check.
* **The uniform path is unchanged.** Same tensor, same labels, no new allocation beyond an index
  list per point; the existing `TestStackScanSensitivities` oracles pass untouched.
* **The bngsim race is still a bngsim bug.** With this ADR the reporter's `trf` fit stops crashing
  in numpy, but on a multi-threaded worker it still fails whenever the race lands on one of the
  three *scored* functions — now with a message that says so. Two fixes are open, neither belonging
  to this ADR: upstream, make `sympy_to_c`'s printer thread-local rather than a mutated module-level
  singleton; here, decide whether the default `Client()` should pin `threads_per_worker=1` the way
  the explicit `parallel_count` branch already does, given that the backend is not thread-safe
  (ADR-0065 already routes a scored Newton scan sequentially for the same reason).
* **Both scan backends are covered.** `bngsim_sbml_model` calls the same `pybnf.data` stacker, so
  the `.net` and SBML/Antimony scan paths get the alignment and the diagnostics from one change.

## Alternatives considered

* **Fail fast in the stacker with a good message.** Rejected as the *only* change: it fixes the
  traceback but keeps a transient, unscored-column difference fatal to the fit. The diagnostics are
  worth having, so they are kept — as the report accompanying the recovery, not instead of it.
* **Cache the first-seen column set per model and reuse it for every point.** Rejected: requesting
  a selector that *this* point's backend refused makes that point's `output_sensitivities` raise, so
  a stable-looking request would trade a diagnosable stack error for a dead simulation, and would
  freeze whichever verdict happened to be observed first.
* **Re-run a dose point whose column set came back short.** Rejected: PyBNF would be retrying
  around an upstream race it cannot see, at the cost of a full simulation per occurrence, and the
  point's tensor is already emitted with a NaN sentinel for the refused function — the retry is a
  guess that the next roll is luckier.
* **Serialize sensitivity codegen behind a PyBNF-side lock.** Rejected: it targets a specific
  upstream implementation detail from the wrong repository, and serializes a phase that is
  legitimately parallel. Guessing which backend call to hold a lock around is not a contract.
* **Pad the missing columns with zeros.** Rejected for the same reason as the union, one degree
  worse: a zero derivative is a *plausible* number, so the fit would silently optimize against a
  gradient that claims a scored observable does not respond to any parameter.

## References

* ADR-0046 — the dose-response/`parameter_scan` surface whose scan `Data` this tensor accompanies
  (steady-state default ↔ PEtab `time = inf`, optional `t_end`).
* Issue #476 / #478 — per-dose-point forward sensitivities for the reset-to-seed scan strategies
  (independent, parity steady-state, KINSOL Newton), whose per-point payloads this function stacks.
* Issue #385 / #447 — the `OutputSensitivities` payload and its typed `observable:` / `expression:`
  / `species:` selectors; `pybnf.gradient.assembly._selector_for` for the scored-column error the
  intersection defers to.
* Issue #522 — `_differentiable_expression_names`, the per-`Result` filter that makes a scan's
  column set a per-point fact (added so a piecewise model's refused functions do not kill every
  simulation on the gradient path).
