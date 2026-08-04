# A normalization chain's sidecar is a list of stages, each keeping the values a later transform overwrote, so the gradient folds the chain rather than refusing it (issue #539)

**Status: Accepted and implemented (2026-08-04).** Closes the deferral ADR-0099 (#533) opened
when it turned a silently wrong Jacobian into an explicit refusal: a column put through two or
more **data-level** normalizations (`normalization pStat = floor 0.03, peak`) is now
differentiated, stage by stage, like any other transform chain.

## The problem

`normalization` is a **chain** (ADR-0066): a comma-separated, ordered list of transforms applied
left to right. Five of them — `peak`, `init`, `zero`, `unit`, `floor` — are *data-level*: they
rewrite the column in `Data` before scoring ever sees it. (`scale` is the sixth and is not one:
it is profiled at scoring time, which is why `floor 0.03, scale` — the motivating ADR-0066 chain
— was already differentiable.)

`Data.normalization` was a `{column: NormalizationRecord}` sidecar, one record per column, so a
second data-level transform **overwrote** the first one's record. And each rule in
`gradient/assembly._normalized_sensitivity` was written against two inputs: the *raw* per-row
sensitivities from the #447 tensor, and the *final* values read back from the rescaled column.
For a chain, neither the earlier stage's facts nor its values survived. Since #533 a second
record carried `chained` and the gradient refused by name; before that, half these chains
threaded the last transform's rule over a column an earlier transform had already moved — a
wrong Jacobian with nothing said.

## The decision

**A chain of transforms is a chain rule. Keep the sidecar in chain order and fold it.**

- **`Data.normalization` becomes `{column: [record, …]}`** in chain order.
  `_record_normalization` appends rather than replaces, and `NormalizationRecord.chained` is
  gone: "this column was normalized before" is now `index > 0`, not a flag to keep in sync.

- **Each stage's rule is the one already written.** Every method's closed form is a chain rule
  *in that stage's own inputs* — the values it consumed and the values it produced — and nothing
  about it assumes those inputs are raw:

  | stage | output | rule (`s` = the sensitivities of what this stage consumed) |
  | --- | --- | --- |
  | `floor` | `y_i = x_i + ρ·max x` | `∂y_i = s_i + ρ·s_ref` |
  | `peak` / `init` | `y_i = x_i / N` | `∂y_i = (s_i − y_i·s_ref)/N` |
  | `unit` | `y_i = (x_i − x_b)/N` | `∂y_i = ((s_i − s_b) − sign·y_i·(s_ref − s_b))/N` |
  | `zero` | `y_i = (x_i − μ)/σ` | `∂y_i = (s_i − s̄)/σ − y_i·(∂σ)/σ`, `∂σ = Σ_k y_k(s_k − s̄)/(K−ddof)` |

  So the fold is literal: `_normalized_sensitivity` starts from the tensor accessor and **wraps
  it once per stage**, each wrapper reading the accessor the previous stage returned. It returns
  an accessor with `tensor_sens`'s own `(col, row)` signature, which is what lets a stage call
  its predecessor exactly as the un-chained rule called the tensor. One transform is the
  one-iteration case and threads precisely the rule it always did.

- **What a stage needed and did not have is its own output values.** Only the last stage's
  survive in the column. So a stage whose output the next transform is about to overwrite keeps
  a copy on its record (`NormalizationRecord.values`), handed to it by
  `_record_normalization` at the moment the next stage records — the values that stage is about
  to replace *are* the previous stage's output. A column normalized once — every job in the wild
  — keeps nothing: `values` stays `None`, the values are read from the `Data`, and the sidecar
  costs exactly what it did before.

  Two alternatives were weighed. **Retain the raw column and replay the chain forward** in the
  gradient (the cheapest to describe) means writing each transform's *forward* formula a second
  time, in `assembly.py`, next to the derivative — two implementations of one reduction, free to
  drift, in a module whose whole premise is that `data.py` owns the transform and records the
  facts. **Invert the chain backwards from the final values** is tempting because peak / init /
  unit / floor are each invertible from their record, but `zero` is not: μ is never recorded,
  only σ. Retaining an overwritten stage's output costs the same one array and needs neither.

- **The capture lives in the `normalize_to_*` methods, not in the `Data.normalize` chain
  driver**, so a chain assembled by calling two of them directly is recorded as correctly as one
  spelled in a config. That makes the moment of capture the thing to get right, and two methods
  had to move to get it: `normalize_to_zero` now centres and scales **out of place** (it used to
  mutate the column in place and record afterwards, by which time the column held the *output*),
  and `normalize_to_unit_scale` — the one method that moves the column before it records, since
  it subtracts the baseline for every column first — snapshots what it consumed at the top. The
  arithmetic in both is unchanged.

- **The fold memoizes per (stage, row).** A stage reads its predecessor at rows other than the
  scored one: the reference row a divisor is read from, and for `zero` *every* row. So a `zero`
  mid-chain asks the stage below it for the whole column once per scored row, and that stage
  asks the one below it — without a memo the work multiplies down the chain, per scored row.
  With one, each (stage, row) is computed once per experiment per evaluation. The tensor at the
  bottom of the fold is left unmemoized, so a single transform costs exactly what it did before.

## Scope

**In:** `data.py` (`Data.normalization` as a list per column, `NormalizationRecord.values`,
`_chain_stage_input`, `_record_normalization` appending; `chained` removed; the out-of-place
`normalize_to_zero` and the pre-baseline snapshot in `normalize_to_unit_scale`),
`gradient/assembly.py` (`_normalized_sensitivity` as a fold returning an accessor,
`_stage_sensitivity` memo, `_stage_rule` holding the per-method rules unchanged, the per-column
accessor cache in `_raw_sensitivity_accessor`).

**Out (unchanged):** every normalized value. `Data.normalize` applies the same transforms in the
same order and writes the same numbers, so no fit's objective value moves — this ADR is about
what is *recorded* alongside them. The single-transform gradient is the same arithmetic in the
same order. `scale` (ADR-0099) is untouched: it is not a `Data` transform, records nothing, and
composes with a chain of any length exactly as before. The PEtab export boundary (ADR-0066):
normalization still has no PEtab v2 representation and is still refused there.

**Deliberately out:** nothing in this area remains deferred. A chain is now differentiable
whatever its length or composition; what a chain *means* is unchanged (ADR-0066 keeps
peak/init/zero/unit sim-only and `floor`/`scale` symmetric), and stacking two whole-trajectory
reductions on one column stays an unusual thing to author — it is now merely unusual, not
refused.

## Verification

- **Finite-difference oracle on the assembled gradient**, extended from the five single
  transforms to five chains: `floor 0.03, peak` and `peak, floor 0.03` (the additive offset on
  each side of the divisor it feeds), `init, zero` and `zero, unit` (the method that couples
  every row, on each side), and the three-stage `floor 0.03, peak, zero`. The oracle is a
  central difference of PyBNF's **own** `Data.normalize` applied to the raw column perturbed
  along its sensitivity, so it arbitrates the whole chain against the actual reduction, not
  against a hand-derived composition.
- **A closed-form arbiter for `floor 0.03, peak`**, where the composition is visible: the peak's
  divisor is the *floored* max and its reference term carries the floor's own `(1+ρ)`. Asserted
  positively, and with the negative control — the last stage's rule alone over the raw
  sensitivities (what the pre-#533 path computed) is a different number, and the test says so.
- **A chain built by direct `normalize_to_*` calls** (`peak` then `unit`, the method that
  records after it moves the column) folds to the same finite difference, pinning the capture
  points rather than the chain driver.
- **The sidecar's own contract**: the records come back in chain order, the overwritten stage's
  retained `values` equal the mid-chain column, and the last stage retains nothing.
- **The config path end to end**: `normalization x = floor 0.03, peak` compiles to its two
  ordered pairs *in chain order* (the previous chain test's second element, `scale`, is not a
  `Data` transform and never had an order to get wrong), and applying the compiled grid records
  the two stages in that order.
- The full default suite is green (3588 passed, 20 skipped).

Relevant ADRs: **0099** (the floor and analytic-scale gradients, whose refusal this replaces),
**0066** (the chain grammar and the two primitives), **0053** (the per-observable normalization
transform and its `∂(raw/N)/∂θ` rules, which this reads unchanged one stage at a time). Closes
issue **#539**.
