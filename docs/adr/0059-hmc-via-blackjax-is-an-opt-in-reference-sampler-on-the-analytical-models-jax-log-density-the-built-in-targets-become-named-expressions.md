# HMC via blackjax is an opt-in *reference* sampler on the analytical model's JAX log-density; the built-in targets become named expressions with explicit constants (issue #425)

**Status: Proposed.** PyBNF gains a gradient-based sampler — `job_type = hmc` (blackjax NUTS) —
that runs **only on the analytical / bring-your-own-log-density model** (ADR-0050), never on a
simulator (BNGL/SBML) model. Its purpose is **evaluating PyBNF's own gradient-free samplers**
(`am`, `dream`, `p_dream`, and any future addition) against a reference-grade NUTS on the canonical
stress geometries and on no-closed-form BYO targets. It is feasible — and cheap — because the
analytical objective is already a *sympy expression* (ADR-0050/0035), so the same expression compiles
to a numpy callable for the existing `score`-column path **and** a JAX callable whose gradient
`jax.grad` produces exactly; blackjax supplies the NUTS engine, so PyBNF reimplements nothing. This
ADR **amends the Tier-3 "Out" boundary of [ADR-0050]** for the analytical special case only. As a
supporting decision, the five built-in targets become **named expressions with explicit constants**
(`objective = banana, a = 1, b = 100`) sharing the one engine, with a JSON sidecar retained only for
the matrix/mixture targets.

## Context

[ADR-0050] makes an analytical objective a bring-your-own log-density **Model**: an inline
`expression =` (or a Python `callable =`) is wrapped behind the existing `score`-column seam, so
`objective = score` (`DirectPassObjective`, ADR-0031) reads the cell and the optimizer/sampler run
unchanged. It deliberately scopes **out** gradient-based sampling ("Tier 3"), with two stated
reasons: *"PyBNF's samplers are gradient-free … there is no autodiff dependency (core is numpy/scipy);
adding NUTS is reimplementing Stan's hard part."*

Both premises have since changed for the **analytical** case (and only that case):

1. **Gradients are free here, not the hard part.** `compile_petab_formula` (`pybnf/petab/formula.py`,
   ADR-0035/0036) parses the expression via PEtab's sympy grammar (`petab.v2.math.sympify_petab`) and
   holds it as a **sympy expression** before `sp.lambdify(..., modules='numpy')`. Lambdifying the
   *same* expression with `modules='jax'` yields a JAX-traceable log-density that `jax.grad`
   differentiates exactly (equivalently, `sympy.diff` gives a closed-form gradient). The analytical
   targets (banana, rotated Gaussian, …) are tiny closed forms; their gradients were never the
   obstacle.
2. **blackjax is the engine.** [blackjax](https://github.com/blackjax-devs/blackjax) is a JAX-native
   *sampler library* (not a PPL): given a `logdensity_fn`, it provides research-grade NUTS with window
   adaptation, dual-averaging step size, and mass-matrix adaptation. PyBNF supplies the log-density and
   the priors; it does not reimplement Stan's sampler.

What has *not* changed is the simulator path. ODE-constrained HMC — differentiating through a stiff
solve via bngsim's CVODES forward sensitivities — costs ~1× a solve *per leapfrog step*, and NUTS
takes many leapfrogs per draw; on a real BNGL posterior that is dominated by simply running `am`/`dream`
for the same wall-clock. **Simulator-path HMC stays firmly out** (it is rejected below, not deferred).

The reframing is therefore narrow: HMC earns its place **as a reference/benchmark sampler on the
analytical model**, not as an inference engine and not on simulators. Its value is twofold —
(a) a recognized "what good looks like" yardstick to score `am`/`dream`/`p_dream` on the canonical
hard geometries, and (b) a trusted second opinion on a BYO target with **no** closed-form posterior
(the common case once priors fold in, so the analytical-truth oracle is unavailable). Trust in the
reference is gated on its own diagnostics: PyBNF already has rank-normalized split-R̂ and bulk/tail ESS
(diagnostics), and blackjax reports divergences.

This ADR also settles two ADR-0050 open questions it touches: how the canned targets are named, and
(partly) the expression dependency boundary.

## The decision

### 1. `job_type = hmc` is a sampler that consumes a JAX log-density, bypassing the score-column loop

`hmc` registers in the `sampler` family (`register_fit_type`) alongside `am`/`dream`. Unlike them, it
does **not** dispatch psets through the dask → `execute()` → `score`-column → `DirectPassObjective`
loop. It builds a JAX `logdensity_fn` in-process and hands it to blackjax NUTS. This is the one
sampler that needs the gradient, and the gradient cannot survive the per-pset dask round-trip, so the
in-process path is intrinsic — it is *not* a parallelism regression, because a single analytical NUTS
chain is a tight numeric loop, and multiple chains run as independent blackjax runs.

### 2. The analytical model exposes a dual numpy/JAX log-density from one sympy source

The BYO/analytical `Model` (ADR-0050's `ExpressionModel`, and the generalized `AnalyticalModel`) gains
a `logdensity_jax()` alongside its existing numpy `execute()`. **Both come from the one sympy
expression**: `lambdify(modules='numpy')` feeds the score-column seam (`am`/`dream`, unchanged);
`lambdify(modules='jax')` feeds blackjax. One source of truth, no parallel hand-maintained math. The
structured matrix/mixture targets (below) supply a small hand-written JAX log-density (a quadratic form;
a `logsumexp` mixture) in addition to their numpy form.

### 3. HMC samples in sampling-space `u`; the target needs no extra Jacobian

PyBNF's priors are defined **entirely in the sampling space `u`** (ADR-0010): a `FreeParameter` holds
the `Scale` (`theta <-> u`) and evaluates `prior.logpdf(scale.forward(theta))`. `am`/`dream` operate in
`u`. HMC does the same — it samples `u` with target

```
log π(u) = Σ_i prior_i.logpdf_jax(u_i)  +  ( − NLL( scale.inverse(u) ) )
```

where `scale.inverse` (identity / `10**u` / `exp(u)`) is JAX-traceable and the NLL is the model's
`logdensity_jax`. Because the prior is *defined* in `u`, there is **no change-of-variables Jacobian to
add** — the target is exactly the density `am` already samples, now differentiated w.r.t. `u`. This
keeps HMC and the gradient-free samplers comparable on the *same* posterior (essential for a fair
benchmark).

### 4. The full prior catalog gains a JAX log-density, up front

Every registered family gets a JAX `logpdf_jax(u)` so **any** prior composes with HMC, not just
uniform/normal. The 16 families: `normal`, `uniform`, `laplace`, `cauchy`, `gamma`, `exponential`,
`chisquare`, `rayleigh`, `beta`, `logistic`, `gumbel`, `weibull`, `inv_gamma`, `half_normal`,
`half_cauchy`, `student_t` (plus `NoPrior`, which contributes `0.0`). Most map directly to
`jax.scipy.stats` (`norm`, `cauchy`, `expon`, `gamma`, `chi2`, `beta`, `laplace`, `logistic`, `t`,
`uniform`); the remainder (`rayleigh`, `weibull`, `inv_gamma`, `half_normal`, `half_cauchy`, `gumbel`)
are simple closed forms written by hand. Each JAX logpdf is validated against the existing
scipy-backed `logpdf` as the oracle (they must agree to numerical tolerance), so the JAX path cannot
silently diverge from the sampler-of-record. The mapping is **family-complete**; the *divergence-free
sampling of constrained supports* is a separate concern (next).

### 5. Constrained/truncated supports are the one genuine HMC-specific wrinkle

The positive-support families (`gamma`/`exponential`/`chisquare`/`rayleigh`/`weibull`/`inv_gamma`/the
half-* priors), `beta`'s `[0,1]`, and explicit truncation (ADR-0047) are sampled by the gradient-free
methods with **reflecting bounds** — a Metropolis device with no HMC analogue. NUTS at a `-inf` wall
diverges. The fix is an **unconstraining bijection** so HMC samples an unbounded `z` and the target
gains the change-of-variables term `log|b'(z)|`; `b(z)` lands strictly inside the open support for
every finite `z`, so the `-inf` wall (and the divergence) is unreachable.

**Resolved (shipped): the full bijection layer, keyed on `support()`.** Rather than a partial first
pass that errors on the constrained families, HMC ships all three standard transforms at once
(`pybnf/priors/bijector.py`): a finite-endpoint classification of `support()` selects identity
(`(-∞,∞)`), `u = lo + exp(z)` (`(lo,∞)`, `log|b'| = z`), `u = hi − exp(z)` (`(-∞,hi)`), or
`u = lo + (hi−lo)·sigmoid(z)` (`(lo,hi)`, `log|b'| = log(hi−lo) + logσ(z) + logσ(−z)`). The transform
is a property of the support *shape*, not the family — so one support-keyed module (the `truncated.py`
ethos) covers the whole catalog with no per-family code, completing item 4's densities to *samplable*.
The **log-scale half** ships with it: `Scale.inverse_jax` (identity / `10**u` / `exp(u)`) makes the
`u → θ` map JAX-traceable, so a `lognormal_var` (normal prior in `u`, identity bijection) and a
`loguniform_var` (uniform box in `log10`-`u`, box bijection) sample cleanly. Diagnostics and the
samples file report `u` (the gradient-free samplers' coordinate; the recorded `Ln_probability` is
un-Jacobianed to `log π(u)`), and PyBNF's R̂/ESS are rank-normalized, hence invariant to the monotone
`z ↔ u` map — so the unconstrained reparameterization is invisible to the comparison machinery.

### 6. The built-in target menu becomes named expressions with explicit constants

This refines ADR-0050's in-scope item *"promote the five built-in targets … (they become canned
expressions / a keyword target)"* into a concrete surface, and resolves its "how the canned targets
are named" open question. A menu target is a **named, pre-packaged expression**, not a closed JSON
enum:

- **Simple targets** (banana, axis-aligned gaussian) — a named built-in with **explicit constants on
  the objective line**, mirroring the existing `noise_model = <family>, <param> = <verb> <arg>` grammar
  (ADR-0021/0031):

  ```
  objective = banana, a = 1, b = 100      # constants optional; defaults documented and echoed at run start
  ```

- **Bring-your-own** — `objective = expression` + `expression = …` (ADR-0050), the user's own math.

- **Structured targets** (`rotated_gaussian` with a full covariance matrix; `multimodal` with a list of
  modes) — a covariance matrix / mode list does not fit a config line or a scalar expression, so these
  **keep a JSON sidecar** (`objective = rotated_gaussian` + a `.target`), carrying a small hand-written
  JAX log-density.

Across all forms, **coordinates bind by name** (the user writes `x1`, `x2`, declared as `uniform_var` /
`parameter:`), per ADR-0050 §4 / ADR-0034 — eliminating `AnalyticalModel._get_param_values`'s
sorted-positional binding (a silent footgun: reorder/rename and the geometry shifts). Constants are
**explicit and echoed** at run start ("banana with a=1.0, b=100.0"), eliminating the silent
`.get('b', 100.0)` default. These two fixes are the discoverability gap #425 names, closed for the menu.

### 7. HMC reuses the existing output and diagnostics surface

HMC writes its draws in the **same samples format** the gradient-free samplers emit, so the entire
posterior-analysis stack works unchanged: the ArviZ `InferenceData` bridge (ADR-0055), LOO/WAIC
(ADR-0056), and the rank-normalized split-R̂ / bulk-tail ESS diagnostics. This is precisely what makes
HMC usable *as a benchmark reference* — its output drops into the same comparison machinery the samplers
under evaluation use.

## Mechanics (proposed)

- **`pybnf/algorithms/samplers/hmc.py`** — a new `Algorithm`/sampler `@register_fit_type('hmc')` in the
  `sampler` family. It resolves the analytical model's `logdensity_jax`, composes the `u`-space target
  (item 3) with the JAX priors (item 4), runs blackjax NUTS (window adaptation), and writes draws in the
  standard samples layout (item 7). Guard: an `hmc` job whose model is a simulator (BNGL/SBML) raises a
  pointed error — *"job_type = hmc requires an analytical/expression objective; a simulator model
  provides no usable gradient (see ADR-0059)."*
- **Model layer** — `logdensity_jax()` on the BYO/analytical models (item 2): expression targets
  lambdify to JAX from the shared sympy expression; structured targets carry a small JAX impl.
- **`pybnf/priors/`** — a `logpdf_jax(u)` per family (item 4), oracle-checked against the scipy `logpdf`;
  the support-aware unconstraining bijection (item 5, shipped as `bijector.py`, keyed on `support()`)
  and the JAX-traceable `Scale.inverse_jax` for log-scaled parameters.
- **`pybnf/config.py` / `pybnf/parse.py`** — the named-target-with-constants objective grammar (item 6,
  reusing the `noise_model` field grammar); constants echoed at run start.
- **Dependencies** — `jax`, `jaxlib`, `blackjax` are a **new optional extra `pybnf[jax]`** (mirroring
  bngsim's `[jax]`); the expression backend is `pybnf[petab]` (sympy). The core stays numpy/scipy
  (ADR-0019); a missing extra raises a pointed "install `pybnf[jax]`" `PybnfError`, never a bare
  `ImportError` (the house pattern). HMC therefore needs both extras present.

## Scope

**In:**
- `job_type = hmc` — blackjax NUTS on the analytical model's JAX log-density, in-process (items 1, 3).
- A dual numpy/JAX log-density from one sympy expression; structured targets carry a JAX impl (item 2).
- The full 16-family prior catalog mapped to JAX `logpdf_jax`, oracle-checked (item 4).
- Menu targets as named expressions with explicit, echoed constants; JSON only for matrix/mixture
  targets; bind-by-name throughout (item 6).
- HMC output reuses the samples format → ArviZ / LOO-WAIC / R̂-ESS (item 7).
- Tests: HMC recovers the **closed-form** posterior moments on `gaussian` / `rotated_gaussian`
  (analytic-truth oracle); HMC vs `am`/`dream` on `banana` / `multimodal` as the sampler-evaluation
  comparison (HMC the reference where its own R̂/ESS/divergences pass); a BYO `expression` sampled by
  HMC; each family's `logpdf_jax` matches its scipy `logpdf`.

**Out (rejected, not deferred):**
- **Simulator-path HMC** (HMC on a BNGL/SBML posterior via bngsim CVODES forward sensitivities).
  Rejected: ~1×-a-solve gradient *per leapfrog* × many leapfrogs per draw is dominated by running the
  gradient-free samplers for the same budget; it would not help anyone. The bngsim JAX
  `differentiable_solve` bridge is real but is **not** adopted here.
- HMC as a **general/production inference engine** or a default `job_type`. It is an opt-in reference
  for sampler evaluation; the gradient-free samplers remain PyBNF's inference path (they are what works
  on the simulator, PyBNF's actual domain).

**Out (deferred — unchanged from [ADR-0050]):**
- Hierarchical / multilevel *sugar* (Tier 2) and a `.stan`-style modelling DSL. True Stan-scale
  hierarchical funnels remain a Stan/PyMC/NumPyro job; PyBNF captures the *engine* for the analytical
  special case, not the *language*.

## Open questions

- **Constrained-support bijection (item 5):** ~~whether the first pass restricts to unbounded + box
  priors and errors on the rest, or ships the bijection layer immediately.~~ **Resolved — ships the
  full layer immediately** (item 5 above): a support-keyed `bijector.py` (log / logit / box-sigmoid,
  with the log-scale `Scale.inverse_jax` half), so every family in the item-4 catalog samples
  divergence-free, not just unbounded + box. The transform keys on `support()` finiteness, so it
  needs no per-family code.
- **blackjax adaptation defaults:** window length, target acceptance, diagonal vs dense mass matrix,
  number of chains/warmup — pick Stan-like defaults; expose a minimal knob set as method config keys
  (ADR-0013 schema).
- **Exact named-constant grammar (item 6):** reuse the `noise_model` field grammar verbatim vs. a
  dedicated objective-line parser; how a structured target's `.target` coexists with the objective line.
- **Where `logdensity_jax` lives:** a method on the model vs. a separate compile pass that produces a
  `(numpy_fn, jax_fn)` pair from the one sympy expression.

## Boundaries (in code — the seams this builds on / where new surface lands)

- `pybnf/algorithms/samplers/basic_mcmc.py` (`lnlikelihood = -score`, `ln_prior`) — the `u`-space
  posterior assembly HMC mirrors in JAX; **the gradient-free path is unchanged**.
- `pybnf/analytical_model.py` / ADR-0050's `ExpressionModel` — gains `logdensity_jax` (item 2).
- `pybnf/petab/formula.py` `compile_petab_formula` — the sympy expression the JAX log-density lambdifies
  from (`modules='jax'`).
- `pybnf/priors/` (`base.py` `Prior`/`FrozenPrior`, the 16 family files, `scale.py`, `truncated.py`,
  ADR-0010/0047) — the `logpdf_jax` mapping and the support-aware bijection seam (items 4, 5).
- `pybnf/algorithms/__init__.py` / `register_fit_type` — `hmc` joins the `sampler` family.
- ArviZ bridge / LOO-WAIC (ADR-0055/0056) and the diagnostics — HMC's output target, **unchanged**.
- `pybnf/config.py` / `pybnf/parse.py` — the named-target objective grammar and constant echo (item 6).

## Consequences

- PyBNF gains a **reference-grade NUTS** for evaluating its own samplers — on the canonical stress
  geometries (with analytic truth where it exists) and on no-closed-form BYO targets (where HMC, gated
  on its diagnostics, is the only trustworthy yardstick) — at the cost of one sampler module, a
  `logdensity_jax` per model, a JAX logpdf per prior family, and an optional `pybnf[jax]` extra. Every
  other piece (priors, posterior assembly, samples output, ArviZ/LOO/diagnostics) already exists.
- It captures Stan's *sampler engine* for the analytical special case (blackjax NUTS + JAX autodiff +
  the existing diagnostics ≈ Stan's engine) **without** Stan's *modelling language* and **without**
  claiming the simulator regime where PyBNF's gradient-free samplers remain the right tool. The
  comparability is deliberately confined to where it serves sampler evaluation.
- The menu's two discoverability footguns — silent geometry defaults and sorted-positional coordinate
  binding — are closed: constants are explicit and echoed, coordinates bind by name.
- **Amends [ADR-0050]:** its Tier-3 "Out" boundary is narrowed to *simulator-path* gradient sampling;
  gradient-based sampling on the *analytical* model is now in scope via this ADR. ADR-0050's Tier-2
  (hierarchical sugar) and its `score`-seam design are unchanged.
- Relates to: **#425** (the analytical-objective surface this extends), [ADR-0050] (the BYO log-density
  model amended here), ADR-0031 (the `score`/objective surface), ADR-0010 (priors in sampling space
  `u`), ADR-0047 (truncation/bounds the bijection must respect), ADR-0034/0043 (bind-by-name /
  `parameter:` record), ADR-0035/0036 (the sympy expression backend the JAX log-density reuses),
  ADR-0021/0031 (the `noise_model` field grammar the named-constant surface mirrors), ADR-0055/0056
  (the ArviZ / LOO-WAIC output surface HMC reuses), ADR-0019 (dependency-free core; `jax`/`blackjax`
  is an optional extra).

[ADR-0050]: 0050-analytical-objective-is-a-byo-log-density-model-expression-and-callable-desugar-to-the-score-column-seam.md
