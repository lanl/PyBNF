# Under edition-2 Mechanism A, PyBNF simulates only the consumed (action, condition) diagonal, not the full cross-product (issue #484)

**Status: Accepted (implemented 2026-07-18).** Under edition-2 **Mechanism A** (ONE
model + `condition:` perturbations, PEtab-aligned) a model runs every synthesized
action under every condition mutant — the full `{action} × {condition}`
cross-product — but only the **diagonal** (each experiment under *its own* condition)
is ever scored. This records that diagonal (plus constraint/postprocessing consumers)
as a per-model **emit-set** and has the backend `execute` skip every other pair, so
**N** experiments × **M** conditions cost **N** simulations instead of **N×(M+1)**.

## The gap

`condition:` (ADR-0028) adds *every* named perturbation as a mutant on the one base
model (`config._load_conditions`), while an `experiment:` registers only the *specific*
`(action, condition)` pair it scores in `exp_data`/`mapping` (`config._load_experiments`).
Each backend's `execute` then blindly expands `actions × mutants` and keys each output
by `action_suffix + mut.suffix`, and the objective quietly drops every suffix not in
`exp_data` (`objective.evaluate_multiple`). So for the Miller et al. 2026 MEK-isoform job
(N=5 experiments, M=4 cell-line conditions) each objective evaluation ran 5×5 = 25
simulations to obtain 5 scored series and discarded the other 20 — an overhead that grows
multiplicatively with both counts.

This is the root cause behind #483, where the wasted off-diagonal suffixes surfaced as an
`am` `output_trajectory` `KeyError` (the sampler allocated trajectory buffers only for the
scored diagonal but the write loop iterated every raw suffix). That crash was fixed
defensively (`ae21d212`); the redundant *work* remained, on every job type.

## The decision

### A per-model emit-set names the pairs some consumer actually reads

`Configuration._compute_emit_suffixes` records, per model, `emit_suffixes[name]` — the set
of *full* output suffixes (`action suffix + condition suffix`) any consumer reads. It is the
union over the three statically-known channels:

* the **scored objective** — the `exp_data` data-keys (the diagonal);
* **constraints** — each `ConstraintSet`'s home `base_suffix` (which covers a
  constraint-only experiment whose data-key is not in `exp_data`) plus any producible
  dotted `suffix.observable` cross-reference;
* **postprocessing scripts** — their `(model, suffix)` targets (a hard direct index in
  `Result.postprocess_data`).

PEtab export reads nothing from `res.out` (it is config-only), and `output_trajectory`
buffers are keyed by the diagonal `time_length` keys — a subset — so neither adds a
requirement.

### The emit-set is distinct from, and a superset of, `_scored_suffixes`

The gradient path's `_scored_suffixes` (ADR/#475) gates *sensitivity computation* per action
and is `exp_data` only. `emit_suffixes` gates *whether a simulation runs at all* and is the
broader `exp_data ∪ constraints ∪ postproc`. The two are kept separate with the invariant
`emit_suffixes ⊇ _scored_suffixes`: a constraint-home suffix must be *simulated* (in `emit`)
but is *not* a scored gradient target (not in `scored`, so it correctly runs sensitivity-free).

### The backend `execute` skips any pair not in the emit-set

`Model._emit_skip(action_suffix)` returns `True` when `emit_suffixes` is set and
`action_suffix + _emit_context_suffix` is absent from it — where `_emit_context_suffix` is
`''` for the base wildtype run and the mutant's suffix on its copy (mirroring the gradient
path's `_sensitivity_offset`). Each bngsim backend consults it: the SBML backend's single
`mutants × actions` loop skips inline on `act.suffix + mut.suffix`; the net/nf backends skip
in the per-action loop and additionally skip building a mutant model at all when no action
pairs with it. Only a **registered** action suffix (an experiment output, in
`model.suffixes`) is prunable — an intermediate pre-equilibration phase emits its own
`simulate` under an unregistered `<name>_preequil` suffix and must always run (it carries
state into the measured phase, ADR-0052).

### Pruning is gated on edition and action separability; otherwise byte-identical

`emit_suffixes[model]` is populated only when the model is edition-2, received
experiment-synthesized actions, and its action suffixes are exactly its experiment names
(**separability** — no hand-written `begin actions` block mixed in, so each synthesized
action is its own reset-independent block that can be skipped individually). Any other model
is left out of `emit_suffixes` → `_emit_skip` never skips → the full cross-product runs
exactly as before. So legacy `mutant:` jobs, non-edition-2 jobs, and mixed models are
byte-identical. The emit-set is computed once at config load and attached to the *runtime*
models in `algorithms.base._initialize_models` (not in config: that method rebuilds the
models, dropping config-set attributes — the same reason the gradient path sets
`_scored_suffixes` on the model list), so it rides `copy.copy`/scatter to the workers.

### A load-time invariant guards against a silent drop

If any consumer references a suffix no `action × condition` pair produces, config load raises
a `PybnfError` naming the pair — turning a would-be silent regression into an actionable
error (`emit − model.get_suffixes()` must be empty). A producible-but-off-diagonal dotted
constraint reference is kept alive (defensive), not pruned.

### Backends and the #483 guard

Only the bngsim backends (`net`, `nf`, `sbml`/Antimony — the edition-2 default) prune. The
BNG2.pl and `.net` subprocess paths auto-scope-out: they write one file / run one process and
their `copy_with_param_set` rebuilds a fresh object that drops the attribute, so
`emit_suffixes` arrives `None` there → no-op. A simulate-line-filter for the subprocess path
is a possible follow-up, out of scope for #484. The #483 `output_trajectory` write-guards
degrade from load-bearing to defensive: off-diagonal suffixes are no longer produced.

## Consequences

- N experiments × M conditions cost N simulations per objective evaluation, not N×(M+1);
  the Miller MEK-isoform job drops from 25 to 5.
- Results are unchanged (only unscored pairs are removed), on every job type; the scored
  objective is provably invariant to pruning.
- The #483 write-guard remains as defense in depth, no longer load-bearing.
- A new tutorial slug (`examples/tutorial/47_condition_perturbations`) is the first
  regular-condition cross-product example and the recovery-tier reference for Mechanism A.
