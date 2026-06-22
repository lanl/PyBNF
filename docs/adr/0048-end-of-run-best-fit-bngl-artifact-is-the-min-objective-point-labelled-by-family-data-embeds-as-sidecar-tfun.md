# The end-of-run best-fit BNGL artifact is the recorded min-objective point, labelled honestly by algorithm family; embedded data rides sidecar `.tfun` files (issue #423-adjacent)

**Status: Accepted → implemented.** At the end of every successful new-era (`edition >= 2`)
run, PyBNF writes one stable-named, self-documenting `Results/<model>_bestfit.bngl` per model:
the original model with the winning parameter set wired in and a header comment that records the
objective value and **labels the point honestly per algorithm family** (an optimizer's *best
fit*; a sampler's *maximum-likelihood point*, which is **not** the MAP). When the opt-in
`embed_best_fit_data` key is set, each observable's experimental data is additionally embedded as
a named `tfun()` reference function backed by a sidecar `.tfun` file, so the artifact self-contains
its comparison curves. Built in two phases on one branch (the bare labelled artifact first, the
data layer second) so the param-substitution change stays reviewable on its own.

## Context

A finished run already deep-copies the best-fit pset into each model and writes a runnable BNGL
(`Algorithm._copy_best_fit_sims` → `BNGLModel.save_all`, `pset.py`), so *pure parameter
substitution into a `.bngl` is not new*. But that file is named for the **winning pset's run id**
(`<model>_<best_name>.bngl`, e.g. `model_iter5run2.bngl`) — opaque, unstable across runs, and
**silent about what the point means**. There is no objective value in the file and, critically,
no statement of whether the parameters are a least-objective optimum or a posterior summary.

That last point is a real trap. **`Trajectory.best_fit()` is the recorded minimum-objective
point for every algorithm, samplers included.** The trajectory is fed `res.score` — the bare
objective / negative-log-likelihood — by `Algorithm.add_to_trajectory` (`base.py`), and
`best_fit()` returns `max(...)` over `-score`. For a Bayesian sampler (`am`, `mh`, `pt`, `dream`,
`p_dream`) the prior is folded in **only** inside the sampler's `got_result`
(`lnposterior = lnprior + lnlikelihood`) to drive the accept ratio; it is **never recorded against
the pset**. So `best_fit()` for a sampler is the **maximum-likelihood (MLE)** point, **not** the
MAP. The two coincide only under flat/box priors and diverge under the informative priors PyBNF
supports (normal, one-/two-sided truncated — ADR-0020/0047). The true per-sample `lnposterior`
*is* available (`Results/samples.txt`'s `Ln_probability` column), so a real MAP is recoverable —
but it is a *different* pset than the min-objective one whose simulations were just copied to
`Results/`.

## The decision

**Emit the recorded min-objective `best_fit()` pset — the same point already copied to
`Results/` — and label it honestly per family, rather than computing a separate MAP.**

- An **optimizer** artifact's header reads *best fit (minimum objective)*.
- A **sampler** artifact's header reads *maximum-likelihood point (minimum recorded objective);
  NOT the MAP — the prior is not folded into the recorded objective. For the posterior mode, take
  the max `Ln_probability` row of `Results/samples.txt`.*

This was chosen over (a) re-ranking the in-memory trajectory by `ln_prior(pset) − objective` to
synthesize a MAP, and (b) reading the MAP from `samples.txt`. Both produce a *different* pset than
the min-objective one whose gdat/scan outputs `_copy_best_fit_sims` already wrote to `Results/`, so
the `.bngl` would no longer match its sibling simulation files — a quiet inconsistency worse than
an honest label. The label is correct in every case; under flat priors MLE *is* the MAP, and under
informative priors the header names exactly where the difference lives and where to find the mode.
(The discriminator is `isinstance(self, BayesianAlgorithm)` — the family base every sampler
already subclasses.)

### Edition scope: new-era (`edition >= 2`) only

The artifact is written **only** when the resolved edition is modern
(`edition.is_modern(...)`, ADR-0031). New-era is where the stable-name/self-documenting surface
belongs, and `edition >= 2` is the contract that assumes the bngsim backend (ADR-0034) the data
layer's `.tfun` round-trip relies on. Legacy runs are byte-for-byte unaffected — they keep only
the existing `<model>_<best_name>.bngl`. The artifact is unconditional within new-era (it is a
strict, near-free improvement over a file already being written); only the *data embedding* below
is opt-in.

### The data-embedding contract: sidecar `.tfun` only, opt-in, time-indexed

When `embed_best_fit_data = 1`, for each `(experiment, observable)` in the model's `exp_data` the
run writes a sidecar `<model>_bestfit_tfun/<experiment>__<observable>.tfun` and a reference
function `expt_<experiment>_<observable>() = tfun('<rel-path>', time)` injected into the model's
`begin functions` block (merged into an existing block, or a new standalone block before
`end model` / `begin actions`). The data is thereby in the model namespace as a comparison curve;
wiring it into gdat output (an observable, a print) is deliberately left to the user, so the
embedded artifact's *simulation semantics are unchanged*.

**Sidecar `.tfun` is the only embedding form — the inline `tfun([t...],[y...],time)` array form is
not used.** The task framing imagined an inline-vs-sidecar size threshold, but verification found
the inline-array form is **neither exercised nor referenced anywhere** in PyBNF or its tests
(`grep tfun(\[` finds only a docstring), while the file-ref form is the one the bngsim bridge test
covers *and* the one the existing staging machinery (`_stage_and_rewrite_tfun_files`, `pset.py`)
is built around. A sidecar-always rule needs no threshold, reuses the tested path, and keeps every
embedded curve auditable as a plain two-column file. Each `.tfun` carries the required
`# <indvar> <obs>` header and **strictly increasing** index column (the experimental rows are
sorted by time and de-duplicated, keeping the first value at a repeated time); `_SD` columns and
the independent-variable column are not embedded.

Embedding is restricted to **time-indexed** experiments (`exp_data.indvar == 'time'`). A
parameter-scan / dose-response experiment's independent variable is a swept parameter, not `time`,
so a `tfun(file, time)` reference would misrepresent it; such experiments are skipped with a log
note rather than embedded wrong (a future ADR can index a scan curve by its swept parameter).

## Mechanics

- **Hook.** `Algorithm._emit_best_fit_bngl(best_pset, best_name)` is called from `run()`'s tail
  (`base.py`), after `_copy_best_fit_sims` / `_rerun_best_fit_to_save_data` and before
  `_finalize_backup_pickle` / `_teardown_sim_dir`. It no-ops off new-era and off a missing best
  fit, so it cannot perturb a legacy run or a run with no successful evaluation.
- **Rendering reuses the ADR-0034 path.** Each model is `copy_with_param_set(best_pset)` then
  `model_text()`; legacy `__FREE` injection and new-era bind-by-id overrides are already handled
  there. Pre-existing `tfun` file refs in the source model are staged with the same
  `_stage_and_rewrite_tfun_files` `save()` uses (run *before* the embedded refs are injected, since
  the new sidecars live in `Results/`, not the source dir).
- **Objective value** comes from `Trajectory.best_score()` (the recorded minimum).
- **Config.** `embed_best_fit_data: int = 0` on `GlobalConfig` (and the `parse.py` int-coercion
  list); a no-op in legacy and when unset.

## Scope

**In:**
- A stable `Results/<model>_bestfit.bngl` per model for new-era runs, with a family-labelled
  header carrying the objective value (Phase 1).
- Opt-in `embed_best_fit_data` embedding each time-indexed observable's experimental data as a
  sidecar `.tfun` + reference function (Phase 2).
- Tests: the emitted BNGL re-parses into a `BNGLModel` with the same parameter namespace; the
  header is family-correct (optimizer vs sampler); the edition gate and the no-best-fit/empty-
  trajectory/non-BNGL no-ops hold; the sidecar `.tfun` content is byte-faithful to `exp_data`,
  sorted/de-duplicated and strictly increasing; `_SD`/indvar columns and non-time experiments are
  skipped; and (bngsim-gated, auto-skipped when absent) the real engine's table-function reader
  accepts the exact generated `.tfun` via `add_table_function` — the embedded data round-tripping
  into bngsim. (A full re-simulation of an *embedded* model would require regenerating its network
  and is left to the recovery tier.)

**Out (boundaries):**
- A *true MAP* artifact (re-ranked by posterior, or read from `samples.txt`) — rejected above as
  inconsistent with the copied min-objective simulations; the honest label points to the mode.
- The inline `tfun([...],[...],time)` array form — unverified against the bngsim parser; sidecar
  only.
- Embedding **non-time-indexed** (parameter-scan / dose-response) data — skipped with a log note.
- Wiring embedded curves into gdat output — left to the user; simulation semantics stay unchanged.
- Legacy (`edition < 2`) runs — unchanged; the existing `<model>_<best_name>.bngl` is their only
  best-fit file.

## Boundaries (in code, each pointing here)

- `pybnf/algorithms/base.py` — `_emit_best_fit_bngl` (the hook, the edition gate, the family
  label), `_build_exp_data_tfuns` / `_inject_function_lines` (the Phase-2 data layer).
- `pybnf/pset.py` — `BNGLModel.copy_with_param_set` / `model_text` (ADR-0034 rendering, reused),
  `_stage_and_rewrite_tfun_files` (staging pre-existing refs).
- `pybnf/config_schema.py` + `pybnf/parse.py` — `embed_best_fit_data` (the opt-in key).
- `pybnf/edition.py` — `is_modern` / `resolve_edition` (the new-era gate).

## Consequences

- Every new-era run leaves a self-documenting, stable-named best-fit model alongside its
  simulations — runnable, and unambiguous about whether it is an optimum or a likelihood summary.
- The sampler MLE-vs-MAP trap is named *in the artifact itself*, with a pointer to the posterior
  mode, rather than being silently mislabelled "best fit".
- The data-embedding path reuses the one tested `tfun` form and the existing staging machinery, so
  it adds no new file-format surface.
- See ADR-0034 (bind-by-id rendering this reuses), ADR-0028 (the new-era experiment/exp_data
  surface the data layer reads), ADR-0031 (the `edition` gate), ADR-0020/0047 (the informative
  priors that make MLE ≠ MAP real). Relates to #423.
</content>
