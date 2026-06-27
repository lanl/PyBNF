# The student_t noise family generalizes the engine to a per-parameter source mapping, with an estimated-gated normalizer per parameter (issue #438 item 1)

**Status: Accepted (implemented 2026-06-26).** Closes the **second and final half of item 1** of
#438 (the low-cost Stan-parity shortlist): the **student_t robust-regression noise model**. The
prior half (eight univariate prior families + the three-parameter prior carrier, ADR-0057) shipped
first; with this the whole of item 1 is done. Builds on ADR-0011 (the `NoiseModel` × `SigmaSource`
decoupling — the `(family, source)` engine this generalizes), ADR-0021 (per-observable noise specs),
ADR-0024 (the location axis), ADR-0031 (the modern `noise_model` surface), and ADR-0056 (the
`log_density` seam LOO/WAIC consumes — student_t feeds it like every family). It mirrors ADR-0057's
**trailing-optional carrier extension**: the prior side grew a third scalar `p3`; the noise side
grows from sourcing *one* noise parameter to sourcing a *mapping* of them, with both changes
backward-compatible by construction.

## Why

PyBNF's per-point likelihood engine (ADR-0011/0021) has, until now, scored every observation with a
noise family carrying exactly **one** noise parameter: Gaussian's σ, Laplace's b, NegBinomial's r.
The owning objective sources that one parameter (`fix_at` / `fit` / `read_exp_file` / …) and adds
the family's likelihood normalizer iff the source is *estimated* — the single rule that makes
`chi_sq` (fixed σ, drop `log σ`) and `chi_sq_dynamic` (free σ, keep `log σ`) the same family. The
engine encoded "one parameter" structurally: a spec was a `(NoiseModel, SigmaSource)` pair, and
`_build_noise_spec` rejected any `noise_model` line with `len(fields) != 1`.

Student-t observation noise is the canonical **robust regression** likelihood — the heavy-tailed
sibling that downweights outliers (the noise analogue of the robust *prior* ADR-0057 added). It is
the first family that genuinely needs **two** noise parameters:

- **σ (scale)** — the spread, exactly Gaussian's role.
- **ν (df, degrees of freedom)** — the tail heaviness / robustness knob. Small ν ⇒ fat tails ⇒
  outlier-robust; ν → ∞ ⇒ Gaussian. This is Stan's/PyMC's `student_t(ν, μ, σ)`, and the same ν is
  the third knob ADR-0057's student_t *prior* exposes.

Each should be **independently sourceable** — `fix_at` a constant or `fit` a free parameter — so a
modeler chooses 1 or 2 adjustable noise parameters: a fixed-ν robust fit (just σ free), a
fully-estimated-noise fit (both free, ν then *needs* a prior because the likelihood is nearly flat
in large ν — exactly the `gamma`/`half_*` priors ADR-0057 shipped), or anything between. That is the
generalization this ADR builds.

## What student_t is (the family)

`pybnf/noise/student_t.py`: `StudentT(NoiseModel)`, a location-scale family like Gaussian/Laplace
(`additive_on`, `location`), plus the shape parameter ν. μ is pinned to the prediction as for every
family (the regression structure). With `z = (μ − forward(obs)) / σ` the per-point NLL splits, and
each piece has a precise home (oracle: `scipy.stats.t(df=ν, loc=μ, scale=σ).logpdf`):

| term | value | home | summed when |
|---|---|---|---|
| **data fit** | `((ν+1)/2)·log(1 + z²/ν)` | `data_fit` | always (the parameter-dependent core; depends on σ via z, and on ν) |
| **σ-normalizer** | `log σ` | `param_normalizers()['sigma']` | iff the σ source is *estimated* |
| **ν-normalizer** | `−logΓ((ν+1)/2) + logΓ(ν/2) + ½·log(νπ)` | `param_normalizers()['df']` | iff the ν source is *estimated* |
| **density constant** | `0` | `_density_constant` | never (there is none — see below) |

The key subtlety is the **ν-normalizer's dual nature**, and the per-parameter mechanism resolves it
for free. When ν is **fixed**, that whole block is a constant in the free parameters — the sampler
must drop it (it cancels in every accept ratio), and the `source.estimated == False` rule drops it
automatically (it is `param_normalizers()['df']`, summed only when df is estimated). When ν is
**estimated**, the block is parameter-dependent and *is* summed — which is what keeps the fit honest
(the t-likelihood is otherwise flat in large ν). So student_t needs **no** `_density_constant`: the
"constant when fixed" the Gaussian carries as `½ log 2π` is, for student_t, the ν-block — already
handled by the estimated-gated normalizer, not a separate pure constant. `_density_constant` stays
`0`, and `log_density` (LOO/WAIC) always includes *both* normalizers regardless of estimated-ness, so
it matches `scipy.stats.t.logpdf` exactly (plus the log-scale Jacobian when on a log scale, ADR-0056).

**Log-scale mean has no finite mean (the guard).** On a log scale `base**StudentT` has *no* finite
mean — heavier-tailed than even Laplace's `b·ln(base) < 1` boundary (the t-distribution's MGF does
not exist), so `E[base**X]` diverges for any ν. Only **median** centering is safe on a log scale;
`location = mean` on a log scale raises a clear `PybnfError` (`mean_offset`), copying Laplace's
log-scale-mean precedent (#419). On the linear scale t is symmetric, so mean = median = μ trivially
(offset 0) and both locations work. Config exposes student_t on the **linear** scale only (there is
no `log_student_t` token); the `additive_on` machinery and the guard are kept for architectural
uniformity and to fail loud if a future token reaches for it.

## The engine generalization (the design content)

A noise spec stops being "a family and *a* source" and becomes "a family and a **mapping** of named
sources". Three coordinated, backward-compatible changes:

### 1. The family owns its parameter names and per-parameter normalizers

`NoiseModel` (`base.py`) grows two small seams, both with single-parameter defaults so existing
families are untouched in body:

- **`noise_params`** — a class attribute, the family's noise-parameter names in declaration order;
  the **first is the primary** scalar passed as `noise`, the rest arrive in a trailing `extra` dict.
  `('sigma',)` for Gaussian, `('scale',)` for Laplace, `('dispersion',)` for NegBinomial,
  **`('sigma', 'df')` for StudentT**. This is the single source of truth for a family's parameter
  names — it *replaces* the engine-side `objective._NOISE_PARAM_NAMES` dict (which duplicated this
  knowledge keyed by config token), letting `_build_noise_spec` read the names off the family it
  just constructed. (Not a registry, ADR-0011: the family owning its own parameter names is the same
  "the family owns its math" principle as `data_fit`.)
- **`param_normalizers(noise, extra=None)`** — returns `{param_name: separable normalizer}`, the
  normalizer each noise parameter contributes; the engine adds each iff *that* parameter's source is
  estimated. The base default attributes the family's whole `log_normalizer(noise)` to its single
  primary parameter (`{noise_params[0]: log_normalizer(noise)}`), so Gaussian/Laplace/NegBinomial are
  byte-identical. **StudentT overrides it** to split `log σ` → `'sigma'` and the ν-block → `'df'`.

`data_fit` gains a trailing-optional `extra=None` (the secondary parameters, `{'df': ν}` for
student_t) — a signature-only change on the existing families (their bodies never read `extra`),
exactly as ADR-0057's families gained a trailing `p3=None` they ignore. `nll` / `log_density` thread
`extra` and sum `param_normalizers(...).values()` for the total normalizer (was the single
`log_normalizer`), so the change is invisible to the single-parameter families and complete for
student_t. The optional-with-a-default parameter (ν defaulting to a fixed value) is declared by the
family as **`noise_param_defaults = {'df': 4.0}`** — the *value* lives with the family (it is a
distributional fact: the default tail heaviness), while the engine builds the `ConstantSigma` from
it, keeping source construction out of the pure kernel (ADR-0011's separation).

### 2. A spec is `(NoiseModel, {param: SigmaSource})`

`_build_noise_spec` now returns the family plus a **dict** of sources (was a bare source). A new
`_build_noise_sources` validates each field name against the family's `noise_params`, builds its
`SigmaSource` via the unchanged `_build_sigma_source`, then fills any omitted parameter that has a
`noise_param_defaults` entry with a `ConstantSigma` (ν → 4) and **requires** the rest (σ has no
default — a `noise_model` line must always state its scale). Unknown names, and a single-parameter
family handed a second field, raise the same clear "no parameter '<x>'" error as before. The
`len(fields) != 1` rejection is gone; 1-or-2 fields are now first-class for a 2-parameter family and
still rejected for a 1-parameter one (its second field is an unknown name).

### 3. The objective carries the default as a mapping, backward-compatibly

`_spec_for` returns `(NoiseModel, {param: SigmaSource})` uniformly. The per-observable overrides
already are that shape (built by `_build_noise_spec`). The **class default** keeps the legacy
single-source attribute `sigma_source` untouched on the registered objfuncs (`chi_sq` =
`DataColumnSigma()`, …) and a new `_default_sources()` wraps it as `{noise_params[0]: sigma_source}`;
a multi-parameter whole-fit default (`noise_model = student_t, …`) is passed through a new
`sigma_sources=` constructor argument and stored directly. `eval_point`, `_pointwise_suffix`,
`_check_columns`, and `required_free_noise_params` all iterate the source mapping — `eval_point`
sources every parameter, builds `(primary, extra)`, calls `data_fit` once, and adds each
`param_normalizers()[name]` iff that source is estimated. One loop replaces the single
`if source.estimated` line, and the single-parameter families flow through it with a one-entry map —
byte-identical scores.

## Surface and forks (resolved with the issue author, 2026-06-26)

```
noise_model = student_t, sigma = <source>[, df = <source>]      # whole-fit default
noise_model <obs> = student_t, sigma = <source>[, df = <source>] # per-observable
```

The `noise_model` grammar already admits several `<param> = <source>` fields (ADR-0021), so the
**parser is unchanged** — it already produces `(family, {'sigma': …, 'df': …}, location)`; only the
engine that interprets that tuple generalizes.

1. **ν default = 4** (`StudentT.DEFAULT_DF`). Omitting `df` pins ν at a fixed 4 — the standard robust
   default (Stan/Gelman's 3–7 range): heavy enough to downweight outliers, light enough to stay
   well-behaved.
2. **Exposed through `noise_model` only** — *no* `objective = student_t` catch-all token. The legacy
   `objective`/`objfunc` tokens are frozen synonyms for the historical least-squares family
   (ADR-0031); student_t is new-era, so it lives only on the modern `noise_model` surface, which
   already expresses its two independently-sourced parameters cleanly. σ is always explicit (no
   `_SD`-column default like `chi_sq`); ν defaults.
3. **Parameter names `sigma` + `df`** — Stan's spelling and Gaussian's `sigma`, not Laplace's
   `scale`. The family this is closest to (a Gaussian with a tail knob) names its scale `sigma`.

## Consequences

- One new ~50-line family file, oracled end-to-end against `scipy.stats.t.logpdf` — fixed-ν default,
  free-σ, free-ν, free-both, the log_density LOO/WAIC path, the log-scale-mean guard — plus the
  engine's multi-parameter spec path (two-field parse, ν default fill, per-parameter normalizer
  gating, `required_free_noise_params` over two sources).
- The noise engine is now genuinely *n*-parameter: a future multi-knob family is a single file with
  `noise_params`, a `param_normalizers` override, and (if any) `noise_param_defaults`. The
  duplicated `_NOISE_PARAM_NAMES` table is retired in favor of the families' own `noise_params`.
- No behavior change for any existing family or config: Gaussian/Laplace/NegBinomial gain only a
  declared `noise_params`, an ignored `extra=None`, and the base `param_normalizers` default that
  reproduces their old single normalizer. The legacy objfuncs keep their `sigma_source` class
  attribute, wrapped on read.
- **PEtab interop is unaffected.** PEtab v2's `noiseDistribution` has no Student-t member, so student_t
  is PyBNF-native (like the extended prior catalog of ADR-0057) and simply isn't part of the
  PEtab round-trip; the exporter already refuses noise families PEtab cannot express.
- **Out of scope:** a `log_student_t` config token (the family supports a log scale internally, but
  with no finite mean there and median-only safety, no token is exposed yet); estimating ν without a
  prior (statistically ill-advised — the modeler should pair `df = fit ν__FREE` with one of
  ADR-0057's positive priors, which the engine neither requires nor enforces).
