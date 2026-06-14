# Per-observable noise decouples (family × σ-source); the σ-source kind decides the normalizer (issue #410)

PyBNF applies **one** `objfunc` to a whole fit: a single `ObjectiveFunction`
whose `NoiseModel` (ADR-0011) is a class attribute applied uniformly to every
data column. PEtab v2 specifies noise **per observable** (`noiseDistribution` ×
`observableTransformation` × the `noiseFormula` σ-source are columns of
`observables.tsv`), so faithful import — the observables chunk of #407 — requires
choosing a noise model **per data column**. It also stands on its own merits: a
single fit against heterogeneous data (molecule **counts** via NegBinomial
alongside **concentrations** via Gaussian, or some observables on a log scale and
others linear) is impossible today. This ADR is the enabling engine; #407's
adapter wiring depends on it.

The observation that unlocks this: today's objfunc-subclass identity **conflates**
two independent axes — the *distribution family* (`chi_sq`/`chi_sq_dynamic` are
both Gaussian; `neg_bin`/`neg_bin_dynamic` are both NegBinomial) and the
*σ-source* (a `_SD` data column, a free parameter, or a config constant). ADR-0011
already isolated the family as a pure per-point `NoiseModel` kernel and noted the
σ-source as the wrapper's job. This ADR finishes that split and lifts it per
column.

- **The per-observable unit is a decoupled `(NoiseModel family, σ-source)` pair,
  not a bundled objfunc code.** We pick Option (A) "decouple", not (B) "select an
  existing objfunc per observable". PEtab v2's `observables.tsv` *is* the decoupled
  model (family and σ-source are independent columns); bundling would force the
  #407 importer to reverse-engineer `(family, σ-source)` back out of a code — the
  parallel-table smell ADR-0019 rejected. The engine is a
  `{column: (NoiseModel, SigmaSource)}` override map on the objective, **defaulting
  every column to the global `objfunc`'s pair** — so a config with no per-observable
  entries is byte-identical to today.

- **The σ-source kind decides estimated-ness, which decides the normalizer.** This
  is the load-bearing invariant. ADR-0011 already established that the Gaussian
  normalizer (`+log σ`) is *retained iff the noise parameter is estimated*; this
  ADR makes the σ-source the thing that knows. A **fixed** σ-source (the `_SD` data
  column, or a config constant) contributes only `data_fit`; an **estimated**
  σ-source (a free parameter) contributes the full `nll = data_fit +
  log_normalizer`. The per-point engine is one expression for all five legacy
  likelihoods:

  ```
  term = family.data_fit(prediction, observation, σ)
  if source.estimated:
      term += family.log_normalizer(σ)
  ```

  This *is* the hard-coded `data_fit`-vs-`nll` choice that today lives in each
  subclass's `eval_point` (`chi_sq` → `data_fit`, `chi_sq_dynamic` → `nll`),
  relocated to the σ-source where it belongs.

- **`SigmaSource` is a first-class abstraction with three kinds, dissolving the
  magic strings.** A `SigmaSource` (`pybnf/noise/source.py`) answers two things:
  its per-point `value(...)` and whether it is `estimated`. The three kinds unify
  what `objective.py` hard-codes today:
  - **`DataColumnSigma(suffix='_SD')`** — the noise parameter read per point from
    the exp column `{observable}{suffix}` (`chi_sq`/`lognormal`). `estimated =
    False`. Owns the "column not found" error and the `_check_columns` exemption
    for that column. The `_SD` convention is **preserved** as the default suffix,
    reframed as *the data-column σ-source* (not deprecated); the native surface
    makes the suffix explicit, so a non-Gaussian family can read a non-`_SD` column
    without the Gaussian "standard deviation" misnomer.
  - **`FreeParameterSigma(name)`** — σ a free parameter resolved by name from the
    pset (`chi_sq_dynamic`/`neg_bin_dynamic`). `estimated = True`. The magic names
    `sigma__FREE`/`r__FREE` survive **as the defaults** for the `_dynamic`
    shorthands, but the coupling dissolves: a per-observable spec names its own
    noise parameter, and the source declares its required name for validation.
  - **`ConstantSigma(value)`** — σ a fixed config constant (`neg_bin`'s
    `neg_bin_r`). `estimated = False`.

  The base `evaluate_multiple`'s hard-coded `free_noise_params` pset-scan
  (`setattr(self, attr, p.value)`) is retired into a generic `{name: value}` pset
  map that `FreeParameterSigma` reads — the same magic-string deletion M2.3 did for
  priors. The legacy AttributeError calling-convention branch (constraints in the
  pset position) is preserved.

- **The five legacy likelihoods become thin subclasses carrying a default
  `(family, σ-source)`; one shared engine replaces three `eval_point` bodies.** A
  new `LikelihoodObjective(SummationObjective)` owns the shared per-point engine
  above, its `_check_columns` (exempting `_SD` only for `DataColumnSigma`
  columns), and the per-observable override map. Each code stays a distinct
  registered subclass — so `objective.ChiSquareObjective()`, the positional
  constructors, and the `isinstance` checks the value-oracle tests pin all stay
  byte-green — but now declares only its class-level default pair:

  | code | family | σ-source |
  |---|---|---|
  | `chi_sq` | `Gaussian()` | `DataColumnSigma()` |
  | `lognormal` | `Gaussian(LOG, MEDIAN)` | `DataColumnSigma()` |
  | `chi_sq_dynamic` | `Gaussian()` | `FreeParameterSigma('sigma__FREE')` |
  | `neg_bin` | `NegBinomial()` | `ConstantSigma(neg_bin_r)` |
  | `neg_bin_dynamic` | `NegBinomial()` | `FreeParameterSigma('r__FREE')` |

  Each row reproduces its legacy objfunc exactly: `chi_sq`/`lognormal`/`neg_bin`
  have a fixed source → `data_fit` only; the `_dynamic`s have an estimated source →
  full `nll` (NegBinomial's `log_normalizer` is 0, so `neg_bin_dynamic` reduces to
  `data_fit` too, matching today). The non-likelihood losses
  (`sos`/`sod`/`norm_sos`/`ave_norm_sos`) and the column-joint `kl` are **not**
  `(family × σ-source)` noise models and are untouched — they keep their own
  `eval_point`/`eval_column` and take no per-observable noise.

- **`neg_bin_dynamic`'s `_Cum` differencing stays the wrapper's prediction
  preprocessing — orthogonal to the (family × σ-source) map.** The one remaining
  legacy quirk (a `_Cum` column's prediction is the row-to-row increment) is
  neither family nor σ-source; per ADR-0011 it is the wrapper's
  prediction-preprocessing. It stays a class-level `_prediction` override on
  `neg_bin_dynamic`, not a third public axis. It is column-**name**-driven (`'_Cum'
  in col_name`), so it composes harmlessly with per-observable family/σ-source
  overrides and is byte-identical for non-`_Cum` columns. Generalizing it into an
  explicit, family-independent cumulative→incident prediction transform (it is a
  data transform, only welded to NegBinomial by COVID-forecasting history) is filed
  as a follow-up, **#418**, and deliberately *not* done here — making it fire for
  any family would silently change `chi_sq`'s behavior on a `_Cum` column, breaking
  this ADR's strict-superset guarantee.

- **The native `.conf` surface is a per-observable `noise_model` table layered over
  the global `objfunc` default.** `objfunc = <code>` stays the backward-compatible
  whole-fit default; a new `noise_model` keyword overrides named observables,
  naming the distribution family and, for each of the family's noise parameters, a
  source — so a human can mix families and σ-sources without PEtab:

  ```
  objfunc = chi_sq                                          # whole-fit default
  uniform_var = b_obs2__FREE 0 5                            # a fitted noise param is an ordinary free parameter
  noise_model obs2 = laplace,   scale      = fit b_obs2__FREE
  noise_model obs3 = normal,    sigma      = read_exp_file _SD
  noise_model obs4 = neg_bin,   dispersion = fix_at 10
  noise_model obs5 = student_t, scale = fit s_obs5__FREE, df = fix_at 4   # forward-compat: a 2-param family
  ```

  The line is
  `noise_model <observable> = <family>, <param> = <source>[, <param> = <source>]…`.
  Each parameter is referenced by its **standard statistical name** (`sigma`,
  `scale`, `dispersion`, `df`, …) — to be documented carefully, since names vary
  across fields — so a multi-parameter family (today's are all single-parameter)
  extends by listing more `<param> = <source>` fields with no grammar change. The
  three sources are **verbs**, one per σ-source kind:
  - `fit <name>__FREE` → `FreeParameterSigma`; the parameter is an ordinary free
    parameter declared the usual way (the `__FREE` suffix and its model↔config
    validation are reused). Estimated → keeps the normalizer.
  - `read_exp_file <suffix>` → `DataColumnSigma`; per-point from the exp column
    `<observable><suffix>`. The suffix is **explicit** (conventionally `_SD`),
    dissolving the hard-coded `_SD` magic and its Gaussian-"standard deviation"
    misnomer — a non-Gaussian family can read `<obs>_scale` instead. Fixed → no
    normalizer.
  - `fix_at <num>` → `ConstantSigma`; a numeric literal in the `.conf` (not a
    model-file reference: a fit can map several models, so "the model constant" is
    ambiguous, and a value that should vary is `fit`, not `fix_at`). Fixed → no
    normalizer.

  Family token ∈ {`normal`, `lognormal`, `laplace`, `neg_bin`, …} (`normal` →
  the `Gaussian` kernel; `lognormal` = `Gaussian(LOG, MEDIAN)`, mirroring the
  existing objfunc alias). Parsed to a structural tuple key
  `('noise_model', observable)` — which, being non-string, rides through
  `_is_unused_key` as structural exactly like free-parameter tuple keys (#401),
  needing no `STRUCTURAL_PASSTHROUGH` change. Granularity is per **observable
  name** (= column); PEtab's per-**condition** axis is orthogonal and stays in
  #407's measurements/conditions chunk.

- **`_load_variables`' two hard-coded magic checks generalize to one σ-source-driven
  check.** The `objfunc == 'chi_sq_dynamic'` ⇒ require `sigma__FREE` and
  `objfunc == 'neg_bin_dynamic'` ⇒ require `r__FREE` special-cases collapse into:
  *every `FreeParameterSigma` named by the objective (default spec + overrides) must
  have a matching declared `FreeParameter`*. The objective is built before
  `_load_variables` runs (`config.py` line 203 vs 205), so it exposes its required
  free-noise-param names and the check derives from them — no objfunc-code switch.

- **A global `laplace` objfunc is added (Laplace with a fitted scale), and the
  Laplace kernel lands in `noise/laplace.py`.** PyBNF has Laplace only as a
  *prior*; this closes the noise gap PEtab's `noiseDistribution=laplace` needs.
  The kernel mirrors `gaussian.py`: `data_fit = |prediction − observation| / b`,
  `log_normalizer = log(2b)` (oracle: `scipy.stats.laplace.logpdf`). It carries
  the same axes; Laplace is symmetric, so `MEAN`/`MEDIAN` coincide (only
  `LINEAR`/`MEDIAN` exercised, as Gaussian started). The global `laplace` code
  uses a **free-parameter** scale `b__FREE` — the dynamic analog of
  `chi_sq_dynamic`, not `chi_sq`: a scale is the kind of thing one estimates, and
  being estimated it **keeps the `log(2b)` normalizer**, which is exactly what
  stops the fit from driving `b → ∞` (and `_SD` would be a Gaussian-"standard
  deviation" misnomer for a Laplace scale anyway). A fixed-scale Laplace remains
  reachable per-observable via `read_exp_file`/`fix_at`.

## Considered Options

- **(B) Select an existing objfunc code per observable** (`noise = obs2 neg_bin`).
  Rejected as the engine model: it re-bundles `(family, σ-source)`, so the #407
  importer would have to *invent* a code for each PEtab `(noiseDistribution,
  noiseFormula)` pair and PyBNF would have to take it apart again — exactly the
  reverse-engineering ADR-0019 rejected. It also can't express a per-observable
  *free* σ with its own name. The decoupled pair is the PEtab-native shape.

- **A noise-family registry mirroring the prior-family registry.** Rejected
  (ADR-0011 stands): the `objfunc` registry already maps codes to wrappers, and
  each wrapper imports the one family it needs. A noise registry would exist only
  to generate config keywords from families — but the native `noise` grammar names
  ~4 families with a fixed token list, not the prior package's generated `*_var`
  keyword pairs. Harvest only if a second consumer needs family enumeration.

- **Put the `data_fit`-vs-`nll` choice on the `NoiseModel` family.** Rejected: the
  normalizer's inclusion is *not* a property of the family (it is one Gaussian
  whether σ is fixed or free) — it is a property of whether the σ-source is
  estimated. Placing it on the source keeps "one family, normalizer governed by
  estimated-ness" (ADR-0011) literally true and makes a per-observable free-σ
  Gaussian automatically keep its normalizer.

- **Fold `_Cum` prediction differencing into the per-observable spec as a third
  axis.** Rejected as over-engineering: it is one legacy quirk on one objfunc,
  column-name-driven, and not part of the settled `(family × σ-source)` decoupling.
  A class-level `_prediction` override reproduces it byte-for-byte without widening
  the spec or the config surface.

- **Defer the native `.conf` surface (engine + PEtab-facing only).** Rejected: the
  #410 issue scopes a native config surface, and a `.conf`-only user with
  heterogeneous data is a real case. The surface is a thin grammar branch over the
  same engine the importer will use; building it now also exercises the engine
  before #407 consumes it.

Relevant ADRs: **0011** (NoiseModel = per-point kernel; wrapper owns the σ-source;
normalizer retained iff the noise parameter is estimated — the seam this lifts per
column), **0004** (noise = three orthogonal axes, PEtab-defaulted not
PEtab-bound), **0019** (importer's registry-driven, no-parallel-table ethos — why
(A) over (B)), **0010** (the Prior/FreeParameter magic-string deletion this mirrors
for σ-sources), **0014** (the structural-key path the `('noise_model', obs)` tuple
rides). Follow-up issues: **#410** (this engine), **#407** (the PEtab observables
adapter that consumes it), **#418** (generalize the `_Cum` cumulative→incident
transform, decoupled from NegBinomial).
