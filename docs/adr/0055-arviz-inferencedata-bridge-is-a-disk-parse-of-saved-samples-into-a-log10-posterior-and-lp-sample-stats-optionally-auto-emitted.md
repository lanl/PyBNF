# The ArviZ `InferenceData` bridge: a disk parse of the saved samples into a log10-space `posterior` + an `lp` `sample_stats`, optionally auto-emitted via a config key (issue #438)

**Status: Accepted (implemented 2026-06-25).** Closes **item 3** of #438 (the low-cost Stan-parity shortlist) and
sets up **item 4** (LOO/WAIC) as a purely additive follow-on. Sits beside ADR-0009 (the
R-hat/ESS diagnostics this bridge hands to the ArviZ ecosystem) and ADR-0010/0022 (the
family×scale split that defines the sampling space the posterior is emitted in).

## Why

PyBNF already does the statistically hard parts of Bayesian inference — it runs five MCMC
samplers (`am`/`dream`/`p_dream`/`pt`/`mh`, all subclassing `BayesianAlgorithm`), writes the
chains to `Results/samples.txt`, and computes rank-normalized split-R-hat + bulk/tail ESS
(Vehtari et al. 2021, `pybnf/diagnostics.py`, ADR-0009) — the same conventions as Stan and
ArviZ. What it does *not* do is hand that output to the posterior-analysis ecosystem
(ArviZ/bayesplot/loo): trace/rank/forest/pair plots, `az.summary`, `az.compare`, and (next)
`az.loo`/`az.waic`. #438 frames this as **"low cost, high leverage — mostly a format bridge,
not new statistics."** This ADR records the bridge's shape.

The bridge is a `from_pybnf(...)` → `arviz.InferenceData` mapping. The design space had four
real forks; each was resolved deliberately and is recorded below.

## The four decisions

### 1. Surface — a standalone `from_pybnf()` **and** a config-gated auto-emit (not either alone)

The documented user API is the standalone function:

```python
from pybnf.inference_data import from_pybnf
idata = from_pybnf("my_fit/Results")     # a Results/ dir, or a samples.txt path
```

post-hoc, decoupled, runnable on any past run without re-fitting. **But a function the user
must remember to call is a feature most users never use.** So PyBNF *also* auto-produces the
object: a new config key **`output_inference_data` (int `0`/`1`, default `0` = off)** makes the
tail of `run()` write `Results/inference_data.nc`, exactly mirroring how `_emit_best_fit_bngl`
auto-writes `Results/<model>_bestfit.bngl` (ADR-0048/0054). With it set, the user reopens the
ArviZ object with zero PyBNF code:

```python
idata = az.from_netcdf("my_fit/Results/inference_data.nc")
```

The auto-emit hook calls the **same builder** as the standalone API (see decision 2), so it is
a ~5-line addition in `run()`'s tail plus one schema key, not a second implementation.

### 2. Source — one disk parse of `samples.txt`, for **both** surfaces (not in-memory chains)

Both surfaces build from `Results/samples.txt`, so they produce **byte-identical** objects and
share **one tested code path**. The `Name` column carries `iter<draw>run<chain>`
(`adaptive_mcmc.py`, recovered by the `(?<=run)\d+` / `(?<=iter)\d+` regexes already in
`_chain_index_from_name`); the distinct `run<c>` count is the number of chains
(= `population_size`). The parser groups rows by chain, orders by `iter`, assigns contiguous
draw indices, and truncates ragged chains to the common minimum draw count (ArviZ needs a
rectangular `chain × draw × param` array — the same min-length discipline `split_chains`
already applies), logging a note when it truncates. The `Ln_probability` column becomes
`sample_stats.lp`.

This was the subtle fork, because PyBNF has **two different recorded chains**:

- **`samples.txt`** is the **saved posterior sample**: written only when
  `iteration > burn_in and iteration % sample_every == 0` (`adaptive_mcmc.py:278`) — *thinned*
  (default `sample_every = 100`) and *post-burn-in*. It is also exactly what `credible*.txt`
  and the histograms are computed from.
- The in-memory **`chain_history`** (what `diagnostics.txt` is computed from) is appended
  **every iteration**, *unthinned*, *including burn-in* (`adaptive_mcmc.py:271`).

We bind the bridge to the **saved sample** (the disk file). Consequences, accepted with eyes
open:

- ArviZ recomputes R-hat/ESS on the saved sample, which has ~`sample_every`-fewer draws than
  `diagnostics.txt`. **R-hat is comparable; ESS reads lower** (ESS scales with draw count). This
  is *correct* — the saved sample genuinely has fewer effectively-independent draws — and it is
  consistent with the rest of `Results/`: an ArviZ HDI lines up with `credible95.txt`. The
  thinned-saved-sample vs dense-running-monitor split is **pre-existing in PyBNF**; the bridge
  faithfully represents the saved draws rather than inventing a third chain.
- The alternative — in-memory `chain_history` for the auto-emit path, disk for standalone —
  would make the auto-emitted `.nc` faithful to `diagnostics.txt` but **emit two different
  objects for one run, where the faithful one cannot be regenerated post-hoc**, on two code
  paths. Rejected (see *Considered / rejected*); it stays available as a future additive
  fast-path without breaking this design.

Two mitigations neutralize the ESS-gap surprise: the builder copies PyBNF's own R-hat/ESS
(read from `Results/diagnostics.txt` when present) into `idata` attributes, and the docstring
states plainly that ArviZ recomputes on the thinned saved sample so `az.ess` is lower than
`diagnostics.txt` **by design**; a user who wants denser ArviZ diagnostics lowers
`sample_every` (one lever, still one code path).

### 3. Parameter space — **sampling space** (log10 for log parameters), not natural units

A log-scaled parameter is emitted into `posterior` in its **sampling space** (`log10` via
`FreeParameter.to_sampling_space` / `_param_vec`), the space the sampler actually moves in and
the space `diagnostics.py` already computes R-hat/ESS in (ADR-0009/0010/0022). So ArviZ's
`az.rhat`/`az.ess`/`az.summary` are in the **same parameterization and use the same Vehtari
method** as PyBNF — directly comparable, not a confusingly different linear-space number — and
the trace/density geometry is the better-behaved log geometry the sampler chose. The variable
is named to make the space explicit (e.g. `log10_k`), so a reader is never misled into reading
a log10 value as a natural one. Linear parameters are emitted unchanged.

This needs each parameter's scale, which `samples.txt` does **not** annotate (its columns are
natural/stored values from `values_to_string`). Recovery:

- **Auto-emit path:** the live `self.variables` are in hand — scale is free and robust.
- **Standalone path:** the original `.conf` is copied into `Results/` at run start
  (`pybnf/pybnf.py:257`), so the builder auto-discovers `Results/*.conf` and reconstructs each
  parameter's scale from it. If scale cannot be recovered (no reachable `.conf`, or it no longer
  loads), the builder **falls back to natural-space and warns**, rather than guessing or failing
  — a degraded-but-honest object beats a crash on an archived run.

### 4. Groups — `posterior` + `sample_stats` only (defer `log_likelihood`/`prior`/`observed_data`)

v1 populates exactly two groups: `posterior` (one variable per fit parameter, dims
`chain × draw`) and `sample_stats` (the `lp` log-posterior column, ArviZ's conventional key).
That delivers the entire trace/rank/forest/pair/`az.summary` surface immediately and keeps the
bridge a pure format mapping. Deferred:

- **`log_likelihood`** — the group `az.loo`/`az.waic` consume. It needs *pointwise* log-lik
  from the per-observable noise engine (`pybnf/noise/`, ADR-0011), which `samples.txt` does not
  carry. This is #438 **item 4** and rides on this bridge as a follow-on (the natural two-step
  arc), not a v1 obligation.
- **`prior`** (draw from the prior families) and **`observed_data`** (pull in the `.exp`
  measurements) — both reach into the prior/data layers, i.e. *new statistics*, past a format
  bridge. Deferred with item 4.

## Surface

- **New module** `pybnf/inference_data.py` exposing `from_pybnf(source, *, config=None,
  variables=None)` → an arviz container (`InferenceData` on 0.x, `DataTree` on 1.x). `source` is a
  `Results/` dir or a `samples.txt` path; `config` is an optional explicit `.conf` override for
  the standalone scale-recovery (auto-discovered from the `Results/` dir otherwise); `variables`
  is the in-process fast path the auto-emit hook uses.
- **New config key** `output_inference_data: int = 0` (mirrors the `output_*` / `*_best_*`
  toggles), with a `docs/config_keys.rst` entry. A `run()`-tail hook emits
  `Results/inference_data.nc` when it is `1`.
- **New optional extra** `arviz` in `pyproject.toml [project.optional-dependencies]`, mirroring
  the `petab` extra: a **lazy import** inside `from_pybnf` with a clear
  `pip install pybnf[arviz]` error, never a hard import (core stays dependency-free, ADR-0019).
  netCDF output needs an xarray engine, so the extra carries an `h5netcdf` (or `netcdf4`)
  backend; the in-memory standalone return needs only `arviz` itself.
- **Both arviz major lines, uncapped.** arviz 1.x (2026) is a rewrite onto xarray's
  `DataTree` that supersedes the 0.x `InferenceData` and changes `from_dict`'s signature. The
  bridge supports **both** rather than pinning to either: the entire API difference is the one
  `from_dict` construction call (0.x takes per-group keywords → `InferenceData`; 1.x takes a
  single group-keyed mapping → `DataTree`), branched in `_build_idata` on the call signature.
  Every downstream access (`.posterior` / `.sample_stats` / `.attrs` / `to_netcdf`) is identical
  across both, verified by running the bridge's full test suite against 0.23.4 **and** 1.2.0. So
  the extra is uncapped (`arviz>=0.17`) — `pip install pybnf[arviz]` never downgrades or
  conflicts with a user's existing arviz, and the bridge is neither stranded on the superseded
  0.x line nor forced onto the brand-new 1.x. Run metadata is stamped on the posterior group
  after construction (not via `from_dict`'s `attrs=`, which 1.x reinterprets as *per-group*
  attrs).
- **Test tier** `@pytest.mark.arviz` registered beside `@pytest.mark.bngsim` in `pyproject.toml`
  + an `importorskip('arviz')`, so the bridge tests run where arviz is installed and skip
  cleanly otherwise (CI's default leg stays dependency-free).

## Considered / rejected

- **In-memory `chain_history` for the auto-emit path (disk for standalone).** The faithful-to-
  `diagnostics.txt` option. Rejected for v1: it emits **two different-resolution objects for one
  run** (unthinned `.nc` at run-end vs thinned `from_pybnf()` afterward), and the faithful one is
  **unreproducible post-hoc** — worse than one always-rebuildable object. It also couples the
  bridge to live sampler internals and forces it to replicate the burn-in/windowing logic. Stays
  available as a future additive in-memory fast-path; choosing disk now forecloses nothing.
- **Natural-units `posterior`.** Plots in real units and a config-free builder, but ArviZ then
  recomputes diagnostics in a *different (linear) parameterization* than PyBNF's — the "which
  R-hat is right?" trap — and hides the log geometry the sampler used. Kept only as the
  *fallback* when a standalone call cannot recover scale.
- **Emit both `k` and `log10_k`.** Most complete, but doubles the variable count, is config-
  dependent, and is noise for a v1 bridge.
- **`observed_data` + `prior` in v1.** Reaches into the data/prior layers — *new statistics*,
  not a format bridge. Deferred with `log_likelihood` to the #438 item-4 follow-on.
- **Standalone API only (no auto-emit).** Rejected because a feature the user must invoke by
  hand is one most users never reach for; the auto-emit is a cheap convenience over the same
  builder.

Relevant: #438 (item 3 here, item 4 — LOO/WAIC — the follow-on), ADR-0009 (R-hat/ESS, the
diagnostics surfaced), ADR-0010/0022 (family×scale → the sampling space the posterior uses),
ADR-0011 (the noise engine that will feed `log_likelihood`), ADR-0019 (core stays
dependency-free; optional extras gate the import), ADR-0028 (the new-era `edition`/config the
`output_inference_data` key joins), ADR-0048/0054 (the end-of-run artifact precedent the
auto-emit hook mirrors).
