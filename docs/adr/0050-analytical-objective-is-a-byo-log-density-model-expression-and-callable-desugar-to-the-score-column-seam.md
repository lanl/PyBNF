# An analytical / user-defined objective is a bring-your-own log-density **Model**, not a new objective; an expression and a callable desugar to the existing `score`-column seam (issue #425, "Tier 1")

**Status: Accepted.** *As built (#425):* both authoring forms ship behind the `score`-column seam,
fileless. The **expression** form (`objective = expression` + `expression = <PEtab math NLL>`) is a
synthesized `ExpressionModel` (numpy via the sympy `compile_objective_expression` backend; bind-by-
name; ADR-0059 item 2 adds the JAX `nll_jax` so HMC differentiates it). The **callable** form
(`objective = callable` + `callable = module:func` *or* `path/to/file.py:func`) is a synthesized
`CallableModel` resolving `f(params, data) -> float` — both a dotted `importlib.import_module`
and a `spec_from_file_location` file path are accepted, validated at config load (fail fast). The
callable is gradient-free (not JAX-traceable), so `job_type = hmc` rejects it with a pointed error.
All open questions are settled. (1) The expression form **reuses** the `pybnf[petab]` sympy grammar
(one grammar; gated behind the extra). (2) **Data binding** (both forms): experimental data is
declared with a top-level `data = f1.exp, f2.exp` key — valid only with `objective = expression` or
`callable` (a pointed error otherwise) — each `.exp` loaded as one experiment into a **name→`Data`
mapping keyed by file stem**, the data kept *model-side* (decision §1: the NLL is computed in the
Model, not the objective) and travelling with it to the dask workers (numpy-backed `Data` pickles
fine). The two forms differ in how they consume it: a **callable** receives the whole mapping as its
second argument (`f(params, data)` with `data = {'f1': Data, …}` or `None`) and reduces it however it
likes; an **expression** becomes a **per-observation** NLL over the parameters *and* the bound data
columns (`0.5*(y - vmax*x/(km+x))^2` references columns `x`/`y`), which the model evaluates per row
and **sums** over every row and experiment (the `Σ per-point NLL` taxonomy, #424) — data columns are
not coordinates, so `coordinate_order`/`nll_jax` vary only the parameter symbols and close over the
columns as constants, making a **data-bound curve fit fully gradient-based under `job_type = hmc`**.
Arbitrary (non-per-observation) reductions are the callable's job; the expression covers the common
per-observation case. Both forms drop the dummy `.exp` and compose with the prior catalog for free.

A user can fit or sample an arbitrary closed-form objective — a negative
log-likelihood, an engineered cost, an analytical test function — **without a BNGL or SBML model
file**, by declaring the target directly in the config in one of two authoring forms: an inline
**math expression** over the free parameters, or a **Python callable** (dotted entry point). Both
are wrapped behind the *same* seam the existing `AnalyticalModel` already uses: a non-simulator
`Model` whose `execute()` emits a one-cell `score` column, which the existing `objective = score`
(`DirectPassObjective`, ADR-0031) reads straight through to the optimizer/sampler. **No new
objective class, no change to the run loop, no change to any sampler.** This is the first-class
surface #425 asks for, scoped to flat-parameter analytical inference ("Tier 1" of the #425
investigation); hierarchical *sugar* and gradient-based sampling are explicitly out (below).

## Context

PyBNF can *already* fit and sample a closed-form target with no simulator — it just isn't a
user-facing feature. `pybnf/analytical_model.py`'s `AnalyticalModel(Model)` reads a `.target` JSON,
computes a negative-log-likelihood directly from the free-parameter vector in `execute()`, and
returns a `Data` with a single `score` column. `objective = score` / `objfunc = direct_pass`
(`DirectPassObjective`, `pybnf/objective.py`) reads that cell and returns it as the objective. The
Bayesian payoff is automatic: a sampler's `got_result` computes `lnlikelihood = -score` and
`lnposterior = lnprior + lnlikelihood` (`pybnf/algorithms/samplers/basic_mcmc.py:143-144`), with
`ln_prior` summed over the per-parameter prior families (`pybnf/algorithms/base.py`,
`pybnf/priors/`). So **a user-supplied NLL is already a valid likelihood term**, and the full
prior catalog (normal / uniform / laplace / cauchy / gamma / exponential / chisquare / rayleigh,
truncatable, scale-aware — ADR-0010/0020/0047/0038) already composes with it.

Three things keep this test-only rather than first-class, all named in #425:

1. **The target menu is closed.** `AnalyticalModel` hardcodes five enum types (gaussian,
   rotated_gaussian, rotated_quartic, banana, multimodal) in a `.target` JSON. A user cannot supply
   *their own* density — a Hill curve NLL, a logistic-growth cost, a mixture they wrote.
2. **The surface leaks implementation.** `direct_pass` is opaque, and it reads a *magic* `score`
   column produced by a model the user can't author without editing PyBNF.
3. **A dummy `.exp` is mandatory.** The config parser requires `model = banana.target : target.exp`
   even though the analytical target ignores experimental data entirely.

PyBNF also already owns a math-expression compiler: `pybnf/petab/formula.py`'s
`compile_petab_formula` turns a math string into a vectorized numpy callable via PEtab's
sympy-backed grammar (`petab.v2.math`), behind the optional `pybnf[petab]` extra (ADR-0035). So the
"user supplies an equation" half is not green-field either.

## The decision

**Generalize `AnalyticalModel` into a user-authored target, declared in the config, behind the
existing `score`-column seam.** The objective layer and the run loop are untouched; all new surface
lives in the *model* layer and the *config* parser. Concretely:

### 1. The seam is the Model, not a new objective

The objective is `score` (the `DirectPassObjective` successor, ADR-0031) in every case. A
bring-your-own (BYO) target is a `Model` subclass whose `execute()` evaluates the user's density at
the current pset and emits the one-cell `score` column, exactly as `AnalyticalModel.execute()` does
today. This is chosen over a parallel "objective = callable" path because the run loop threads the
**pset into the model** (`copy_with_param_set`) and the **sim_data into the objective**, never the
pset into the objective; computing the NLL model-side is therefore the seam that exists, requires no
new plumbing, and parallelises for free (each pset evaluation is dispatched as a model "run" through
the same dask path). The `score` column contract (ADR-0031) is the boundary both forms cross.

### 2. Two authoring forms, both desugaring to that seam

- **Expression form** — an inline math expression over the free-parameter names (and, optionally,
  data columns), e.g.

  ```
  objective   = expression
  expression  = 0.5*((1 - x1)**2 + 10*(x2 - x1**2)**2)     # banana NLL, no file
  uniform_var = x1 -5 5
  uniform_var = x2 -5 15
  fit_type    = am
  ```

  Recommended backend: reuse `pybnf/petab/formula.py`'s sympy-backed compile path so PyBNF owns one
  math grammar. Because that path is the optional `pybnf[petab]` extra, the **dependency boundary is
  a sub-decision to settle** (see Open questions): either gate the expression form behind that extra,
  or vendor a small dependency-free evaluator (e.g. an `ast`-restricted or `asteval`-style numeric
  evaluator) so the core stays numpy/scipy-only (the project's standing constraint — ADR-0019,
  "core stays dependency-free"). The grammar must **never** be raw `eval`.

- **Callable form** — a dotted Python entry point:

  ```
  objective = callable
  callable  = mymodel:negative_log_likelihood
  ```

  resolved to `f(params: Mapping[str, float], data: Data | None = None) -> float` returning the
  scalar NLL. This is the escape hatch for everything the expression grammar cannot express:
  `logsumexp` mixtures, loops over groups/replicates, `scipy.stats` densities, a hand-rolled
  hierarchical pooling term. It is consistent with PyBNF's existing trust model (the config already
  names model files PyBNF imports and simulators it executes), but the **expression form is the
  documented default** because it imports no arbitrary code.

### 3. No model file, optional `.exp`, priors for free

A BYO objective needs **no `.target`, no `.bngl`, no `.sbml`, and no dummy `.exp`.** The model is
the expression/callable declared in the config; experimental data is bound only when the objective
actually reads it (a Hill curve fit to measurements supplies an `.exp`; a pure analytical target
like banana supplies nothing). Free parameters are declared exactly as today — `uniform_var`,
`normal_var`, the new-era `parameter:` record (ADR-0043) — so **informative-prior Bayesian
inference on an analytical model is `objective = expression` + a `normal_var`/`parameter:` prior +
`fit_type = am`, with zero new sampler code.**

### 4. Convention: the value is an NLL (lower is better)

The user supplies a **negative log-likelihood** (or, for an optimizer, a cost): smaller is better,
the same contract as `direct_pass`/`score` and the rest of the objective taxonomy (#424 — every
objective is an NLL with parameter-independent constants dropped). The sampler negates it
(`lnlikelihood = -objective`), so a user writes the *cost*, not the log-likelihood, and there is no
sign surprise. Parameters are bound **by name** (matching new-era bind-by-id, ADR-0034), not by the
sorted-vector positional convention `AnalyticalModel._get_param_values` uses internally.

### 5. `direct_pass` stays the internal alias

`direct_pass` remains a working developer/internal spelling (the analytical test tier and golden
configs keep using it; #425 explicitly wants to avoid the rename + golden-churn yak-shave).
`objective = score` is the bare passthrough; `expression` / `callable` are the new BYO tokens, added
to `build_named_objective`'s dispatch (`pybnf/objective.py`) the same way `score` is.

## Mechanics (proposed)

- **New model classes** alongside `AnalyticalModel`: an `ExpressionModel` and a `CallableModel`
  (names to confirm), each a `Model` subclass with the same shape as `AnalyticalModel` —
  `copy_with_param_set`, a no-op `save`, `get_suffixes`, and an `execute()` that evaluates the target
  at `self._pset` and returns `{'<suffix>': Data(score column)}`. The expression model holds a
  compiled callable; the callable model holds the resolved entry point.
- **Config / parser.** A model declared by `objective = expression|callable` + its companion key,
  resolved in the model-loading path (`pybnf/config.py`) so no model *file* is required and `.exp`
  is optional when the objective ignores data. The two tokens join `score` in
  `build_named_objective`.
- **Objective layer: unchanged.** `objective = score` → `DirectPassObjective` reads the `score`
  cell. The BYO model produces that cell.
- **Sampler / prior layer: unchanged.** `ln_prior` + `-score` already assemble the posterior; the
  prior families already compose.

## Scope

**In (Tier 1):**
- A first-class, file-free analytical objective via an inline expression and a Python callable,
  desugaring to the `score`-column seam.
- Drop the dummy `.exp` requirement for data-ignoring objectives.
- Promote the five built-in analytical targets (banana, gaussian, …) from test fixtures to a
  documented, user-reachable feature (they become canned expressions / a keyword target).
- Bind free parameters by name; document the NLL (lower-is-better) convention.
- Tests: an expression NLL and a callable NLL each (a) optimise to a known mode and (b) sample to
  known posterior moments, reusing the analytical harness (`tests/integration_harness.py`); a
  data-bound expression fits a closed-form curve (Hill / logistic) to an `.exp`; the no-`.exp`,
  no-model-file config parses; parameters bind by name; informative-prior runs fold the prior in.

**Out (boundaries — deliberately deferred):**
- **Hierarchical / multilevel *sugar*** (plates, partial-pooling notation, hyperprior blocks,
  transformed-parameter blocks — "Tier 2"). A two-level model is *expressible* today by writing the
  group structure and hyperprior terms into the callable form; what is out is *notation* for it. It
  is bounded by the next item and gets its own ADR.
- **Gradient-based sampling (HMC / NUTS / ADVI) — "Tier 3".** PyBNF's samplers are gradient-free
  (adaptive Metropolis, DREAM); there is no autodiff dependency (core is numpy/scipy). Adding NUTS
  is reimplementing Stan's hard part and is out of scope; gradient-free MCMC also mixes poorly on the
  funnel geometry hierarchical posteriors produce, which is why Tier 2 is bounded by Tier 3.
- A standalone modelling DSL / `.stan`-style language. The surface is the existing config + the two
  authoring forms, not a new language.

## Open questions (all RESOLVED as built — see the Status note)

- **Expression dependency boundary:** ~~reuse the `pybnf[petab]` sympy grammar vs. a dependency-free
  evaluator.~~ **Resolved:** reuse the `pybnf[petab]` sympy grammar (one grammar; the analytical path
  gates behind the optional extra, the core stays numpy/scipy).
- **Exact config spelling:** ~~`objective = expression` + `expression =` vs. a dedicated key; how the
  canned targets are named.~~ **Resolved:** `objective = expression` + a companion `expression =` key
  (and `objective = callable` + `callable =`); the canned targets are named inline on the objective
  line (`objective = banana, a = 1, b = 100`; ADR-0059 item 6). A **data-bound expression** references
  `.exp` columns directly (`expression = 0.5*(y - vmax*x/(km+x))^2` + `data = curve.exp`): the column
  headers join the parameters in the compile namespace and the expression is summed per observation —
  the same `data =` key the callable uses.
- **Callable signature & data binding:** ~~the precise `(params, data)` contract and how multi-
  experiment data is presented.~~ **Resolved:** `f(params, data)` where `params` is a name→value
  dict and `data` is a name→`Data` mapping keyed by file stem (`None` when no data is bound),
  declared with the `data = f1.exp, f2.exp` key (valid for `expression`/`callable`) — multi-
  experiment data is keyed by name exactly as parameters are.

## Boundaries (in code — the seams this builds on / where new surface lands)

- `pybnf/analytical_model.py` — `AnalyticalModel(Model)`: the proven non-simulator-`Model` seam the
  new `ExpressionModel` / `CallableModel` generalise (`execute()` → `Data` `score` column).
- `pybnf/objective.py` — `DirectPassObjective` / `build_named_objective`: `objective = score` reads
  the cell; the `expression` / `callable` tokens join its dispatch. **Objective math unchanged.**
- `pybnf/algorithms/samplers/basic_mcmc.py` (`lnlikelihood = -score`) and
  `pybnf/algorithms/base.py` (`ln_prior`): the posterior assembly the BYO objective feeds.
  **Unchanged.**
- `pybnf/priors/` — the prior catalog (ADR-0010/0038) that composes with the BYO likelihood for
  free; `pybnf/algorithms/base.py` `parameter:` record (ADR-0043) for the prior declaration surface.
- `pybnf/petab/formula.py` — `compile_petab_formula`: the candidate sympy-backed expression backend.
- `pybnf/config.py` / `pybnf/parse.py` — the model-loading + `.exp`-optional changes land here.

## Consequences

- PyBNF gains "bring your own log-density" — fit or sample an arbitrary analytical NLL with the full
  optimizer / sampler / parallel / prior machinery, **no BNGL/SBML file** — at the cost of two thin
  `Model` subclasses and a config change, because every hard part (samplers, priors, posterior
  assembly, the `score` seam) already exists.
- It captures Stan's *declarative-modelling convenience* for small/medium analytical models
  (dose-response / Hill, Michaelis-Menten steady states, growth curves, count-noise models — several
  mapping onto the noise families in `pybnf/noise/`) without trying to be Stan's *gradient-based
  engine*.
- It cleanly closes #425's three problems: the closed target menu (now BYO), the leaky
  `direct_pass`/magic-`score` surface (now `objective = expression|callable`), and the dummy `.exp`
  (now optional).
- Relates to #425 (this is its design), #424 (the NLL-with-constants-dropped taxonomy this convention
  follows), ADR-0031 (the `score`/objective surface reused), ADR-0010/0038/0043 (the priors that
  compose for free), ADR-0035 (the candidate expression backend), ADR-0019 (the dependency-free-core
  constraint the expression backend must respect).
