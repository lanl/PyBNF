# The Differential Evolution family's convergence test is an absolute objective range over the finite population, carried by its own key, so a likelihood fit's negative objective no longer stops the run after generation 0 (issue #561)

**Status: Accepted and implemented (2026-08-19). The DE sibling of ADR-0106.** ADR-0106
(issue #550) fixed exactly this reasoning for CMA-ES: "PyBNF minimizes a negative
log-likelihood, which is unbounded below, so `|f|` GROWS as the fit improves." `de` and
`ade` are the unfixed siblings, and their form is worse — a **ratio** of objectives
rather than a relative additive threshold. This ADR makes the DE-family convergence
criterion an **absolute range in objective units**, assessed over the **finite**
fitnesses only, and gives it **its own key**, `de_tolfun`, because a ratio and a range
have no common scale.

## The defect

Both members of the family test convergence with a ratio of the population's objective
values.

`pybnf/algorithms/optimizers/differential_evolution.py`, `de` (per island):

```python
if (np.min(self.fitnesses) != 0) and (np.max(self.fitnesses) / np.min(self.fitnesses) < 1. + self.stop_tolerance):
    return 'STOP'
```

`ade`, without even the `!= 0` guard:

```python
if np.max(self.fitnesses) / np.min(self.fitnesses) < 1. + self.stop_tolerance:
    return 'STOP'
```

The test means "the spread is small **relative to the values**", which is a convergence
statement only when the objective is positive and bounded below by 0 (a χ², an SSE). It
fails two ways otherwise, and both are the **sign** of the objective, not any real
convergence:

1. **All fitnesses negative.** `max/min` lands in `(0, 1]` — for `max = -59.5`,
   `min = -166.0` it is `0.358`, below `1 + stop_tolerance` for any non-negative
   tolerance. So the run stops at the first check regardless of how spread out the
   population is: `0.358` is a population spanning 107 NLL units being called converged.
2. **Any failed simulation, with any negative fitness present.** A failed point scores
   `inf` (`base.py`, `add_to_trajectory`), so `max/min = inf / (negative) = -inf`, below
   *every* threshold. This defeats the obvious workaround: setting `stop_tolerance` very
   negative does not disable the test, because `-inf` is below that too.

That is every estimated-σ likelihood problem — the whole Grein et al. 2026 benchmark
subset-I corpus (23/23 slugs, all with `noise_model` estimating σ and all with a negative
objective at their nominal point) — so **both members of the Differential Evolution
family were unrunnable there**.

### Observed

`Borghans_BiophysChem1997` from the subset-I corpus (23 free parameters,
`noise_model = lognormal, sigma = fit sigma`, so the objective is a negative NLL),
`job_type = de`, `population_size = 400`, `max_iterations = 600`, `islands = 4`, PyBNF
`095a5a14`, bngsim 0.13.0:

| `stop_tolerance` | outcome | generation-0 jobs | generation-1 jobs |
|---|---|---:|---:|
| `1e-6` | `Stop criterion satisfied with objective function value of -59.48` | 125 | **0** |
| `-1e9` | `Stop criterion satisfied with objective function value of -0.78` | 32 | **0** |

Both runs terminate inside the first generation, having spent a 240,000-evaluation
budget on 0 generations of search. The second row is the workaround point:
`stop_tolerance = -1e9` puts the threshold at −1e9 and *still* fires, because failed
points make the ratio `-inf`. There is no configuration that disables this test. Note the
reported "best objective" is *worse* in the second run (−0.78 vs −59.48): with the test
firing mid-generation, what gets reported is whatever the partial population happened to
hold.

## The decision

### 1. The criterion is an absolute range in objective units, over the finite fitnesses

```python
finite = np.asarray(self.fitnesses, dtype=float)
finite = finite[np.isfinite(finite)]
return bool(finite.size and (finite.max() - finite.min()) <= self.de_tolfun)
```

The range `max - min` is **sign-agnostic**: an all-negative population spanning 107 NLL
units has a range of 107, nowhere near any convergence tolerance, so it does not fire.
Ignoring the non-finite entries is what makes a failed simulation unable to either
**satisfy** or **defeat** the test — the same robustness ADR-0106 chose *not* to adopt
for CMA-ES's current-generation conjunct, here adopted because it is the whole point of
the fix. An empty finite set (every member still `inf`, e.g. before the first result) is
never "converged". This also subsumes the original `!= 0` guard against dividing by an
all-zero population, which `ade` never had — a whole population at exactly 0 now reports
converged (range 0) with no division at all, rather than computing `0/0 = nan`.

The check lives in one shared helper, `DifferentialEvolutionBase._population_converged`,
so `de` and `ade` cannot drift apart again (the missing `ade` guard was exactly such a
drift).

### 2. It gets its own key, `de_tolfun`, falling back to `stop_tolerance`

The old `stop_tolerance` was a **dimensionless ratio**; the new threshold is a **range in
objective units**, whose natural magnitude is set by the data, the noise model, and the
number of scored points. These cannot share one well-set value, so — exactly as ADR-0106
split `cmaes_tolfun` off `cmaes_stop_tol` — `de_tolfun` is the range tolerance, and
**unset it falls back to `stop_tolerance`**, so an existing config keeps the threshold
*magnitude* it had. The reinterpretation from ratio to range is unavoidable (it is the
defect), but no config silently acquires a *different number*.

### 3. Island DE assesses convergence only once every island has run

The convergence check reads the global `self.fitnesses` but fires per island, when that
island completes an iteration. Ignoring `inf` removes a coupling the old ratio relied on:
an unevaluated island (all members at the initial `inf` sentinel) used to make the global
`max` infinite and thereby block convergence. Without that, a single island finishing its
first iteration with a collapsed finite subpopulation could stop the whole multi-island
run before the other islands had searched at all.

So the `de` call site gates the check on `min(self.iter_num) >= 1` — every island has
completed at least one iteration, hence no member is still at its initial `inf` (only
failed sims are `inf`, and those are correctly filtered). This restores the "wait for the
whole population" half of the old behavior without the sign / failed-sim half. For a
single island (`islands = 1`, the default) it holds from the first iteration on, so
single-island DE and `ade` — whose per-iteration check point already guarantees a full
population pass — are unaffected.

## What is deliberately not changed

* **The reported reason string.** `de`/`ade` return `'STOP'` and the run loop prints the
  generic "Stop criterion satisfied with objective function value of …"; this is
  unchanged. ADR-0106 added a threshold-reporting reason for the CMA-ES *restart battery*
  because a restart's arithmetic (range vs window vs tolerance) was otherwise
  unrecoverable from the log; a DE convergence stop has no such hidden arithmetic.
* **The default magnitude.** `de_tolfun` unset is `stop_tolerance` (default `0.002`).
  A fit that wants the old *relative* semantics has no equivalent — but that semantics was
  only ever meaningful for a positive objective, where an absolute range at the same
  magnitude is a stricter, well-defined stop.

## Consequences

* **The DE family runs on a likelihood objective.** On the reported fit the run no longer
  stops at generation 0; it searches to `max_iterations` (or a genuine range collapse).
  The whole subset-I corpus becomes reachable by `de`/`ade`, which mattered because on its
  last unsolved slug (wshlavacek/BNGL-Models#38) `de` was reached for precisely because
  `gntr`, `cmaes` and `pso` all converge to the same shallow attractor.
* **A failed simulation neither stops nor blocks the search.** The `inf` is filtered, so
  one dead candidate can no longer end a run (defeat by `-inf`) nor, in island DE, hold a
  finished island open forever.
* **Existing configs keep their threshold magnitude.** `de_tolfun` unset is
  `stop_tolerance`; only the ratio→range reinterpretation changes, and only in the
  direction of a better-defined stop.
* **One new config key.** `de_tolfun` joins the DE-family schema
  (`Optional[float]`, `ge=0`), the `parse.py` float-token list, and the golden
  effective-config corpus, which moves by that one key across the nine DE-family fits.
* **What a user should set.** On a likelihood fit, set `de_tolfun` to the smallest
  population objective spread you still consider unconverged (a range in your objective's
  units); leave it unset on a classic SSE/χ² fit to keep the `0.002`-magnitude stop.

## Alternatives considered

* **Reuse `stop_tolerance` and just drop the ratio.** Rejected for the same reason
  ADR-0106 rejected reusing `cmaes_stop_tol`: a dimensionless ratio and an
  objective-units range have no common scale, so one key cannot be well-set for both, and
  silently redefining the existing key's units is worse than adding one.
* **Require ≥ 2 finite members for a "spread".** Considered. A single finite member
  (range 0) reports converged, which on a population where all-but-one sim failed is a
  degenerate but defensible "one viable point" stop. Not adopted: it is an extra rule for
  a pathological state (a persistently 9/10-failing population is not going to search
  productively either way), and the issue's proposed form uses `finite.size >= 1`.
* **Distinguish an unevaluated `inf` from a failed-sim `inf` by tracking evaluation
  state.** Rejected as heavier than needed: `min(self.iter_num) >= 1` is an exact proxy
  in island DE (every island having run once ⟺ no initial-`inf` members remain) and needs
  no new state.

## References

* Issue #561 (this defect) and the reproduction write-up wshlavacek/BNGL-Models#38
  (`Borghans_BiophysChem1997`, Grein subset-I).
* ADR-0106 (the CMA-ES sibling, #550): the same "unbounded-below likelihood defeats a
  relative/ratio convergence threshold" reasoning, and the `cmaes_tolfun`-off-`cmaes_stop_tol`
  own-key precedent this mirrors.
