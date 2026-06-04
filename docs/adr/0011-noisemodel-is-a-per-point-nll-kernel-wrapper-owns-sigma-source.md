# NoiseModel is a per-point NLL kernel; the objfunc wrapper owns the σ-source (M2.4 build shape)

`objective.py`'s probabilistic members (`chi_sq`, `chi_sq_dynamic`, `neg_bin`,
`neg_bin_dynamic`) entangle three concerns in one flat file: the **distribution
family** math, the **noise-parameter source** (σ/r), and the **per-point
iteration** (the row loop, weighting, NaN/Inf→`None`, ind-var rounding). M2.4
extracts the noise model into a `pybnf/noise/` package as a **behavior-preserving**
change — the contract is the existing green `tests/test_objective_funcs.py`
(by-hand/scipy value oracles for every objfunc) + `tests/test_load_obj_func.py`
(registry dispatch), kept byte-green (extend, don't weaken). Building on ADR-0004
(noise model = three orthogonal axes, PEtab-defaulted not PEtab-bound) and ADR-0005
(self-registering dispatch), and mirroring ADR-0010 (M2.3's `Prior`/`FreeParameter`
split), we settled this shape:

- **`NoiseModel` is a pure per-point NLL kernel** — `nll(prediction, observation,
  noise)` returning a scalar, scale-agnostic, one file per family under
  `pybnf/noise/` (`Gaussian`, `NegBinomial`, …). The existing
  `SummationObjective` harness is **unchanged** and delegates the per-point math
  to the kernel. This mirrors M2.3 exactly: the harness that owns the iteration
  (here `SummationObjective`, there `FreeParameter`) delegates the pure family
  math to a small scale-agnostic object (here `NoiseModel`, there `Prior`).

- **The three axes compose into the kernel as
  `nll = data_fit_term(prediction, observation, θ_noise) + log_normalizer(θ_noise)`.**
  The **distribution family** supplies both terms; the **additive-noise scale**
  is a transform wrapping the family (linear for Gaussian/NegBinomial); the
  **location interpretation** selects which summary the prediction maps to
  (mean for the symmetric current families).

- **The likelihood normalizer is retained iff the noise parameter is itself
  estimated**, and dropped as a parameter-independent constant when the noise
  parameter is fixed. This is *not* a difference between two families: it is why
  `chi_sq` (σ from the `_SD` data column, fixed) omits the `+logσ` term while
  `chi_sq_dynamic` (σ a free parameter) keeps it — one Gaussian family, the
  normalizer governed by whether σ is sampled. `NegBinomial` has no separable
  normalizer (a PMF is self-normalizing), so `neg_bin` and `neg_bin_dynamic` both
  use the full `−logpmf`.

- **The objfunc wrapper owns the σ-source and prediction-preprocessing; it stays
  in `objective.py`.** Each objfunc (`ChiSquareObjective`, … — a
  `SummationObjective` subclass) sources the noise parameter from one of three
  places — per-point data column (`_SD`), free parameter (`sigma__FREE`/`r__FREE`),
  or config constant (`neg_bin_r`) — and carries quirks like
  `neg_bin_dynamic`'s `_Cum` differencing, before calling the family kernel. The
  source's param-ness is what drives normalizer inclusion. The base
  `evaluate_multiple`'s hardcoded `r__FREE`/`sigma__FREE` pset-scan is retired
  into a **declarative noise-source** on the wrapper — sequenced as its own commit
  *after* the behavior-identical extraction, so the magic-string coupling M2.3
  deleted for priors is deleted here too.

- **Two shapes of noise model; only the per-point shape is abstracted now.**
  *Per-point (independent)* noise models (Gaussian, NegBinomial) are the kernel
  above, delegated to by `SummationObjective`. *Column-joint* noise models —
  whose per-observation contributions couple across a data column so the
  likelihood does not factor — are **not** abstracted. `kl` is a genuine
  likelihood (the multinomial NLL, ≡ minimizing KL-divergence), but its per-point
  term `−exp_i·log(sim_i/Σⱼsimⱼ)` depends on the whole column through the
  normalization, so it cannot be the per-point kernel. It stays a plain
  `ColumnSummationObjective`. The column-joint abstraction is harvested when a
  second member (Dirichlet-multinomial; a correlated-error/Gaussian-process
  likelihood) justifies it (ADR-0009's ≥2-user bar) — `ColumnSummationObjective`
  is its natural future home, peer to `SummationObjective` for the per-point kind.

- **Construction is uniform and recipe-free via a `from_config(config)`
  classmethod; the registry drops `config_args` (ADR-0005).** `_load_obj_func`
  calls `entry.cls.from_config(self.config)` for every code; the per-objfunc
  positional recipe in the registry disappears. The classes' `__init__`
  signatures are **preserved** so the value-oracle tests that construct objfuncs
  directly stay byte-green — only `test_load_obj_func.py` (which pinned the now-
  removed recipe) updates to the new contract. The `neg_bin`-requires-`neg_bin_r`
  and `*_dynamic`-requires-`{r,sigma}__FREE` cross-checks stay in `config.py`
  (ADR-0006 #5).

- **The location and additive-noise-scale axes are modeled from the start but
  exercised only at trivial values** (`linear`, `mean`) by Gaussian/NegBinomial,
  then proven by adding **`lognormal = Gaussian × additive-on-log × location-
  median`** — a *reconfiguration of the existing Gaussian family adding zero new
  distribution families*, the way Laplace proved the prior seam (ADR-0010). It is
  asymmetric, so it lights up the location axis (mean ≠ median ≠ mode), and its
  noise is additive on the log scale, so it lights up the scale axis. Location is
  overridable at the object level (one can construct `location=mean`); only the
  PEtab-default `median` is wired to a config objfunc. Value oracle:
  `scipy.stats.lognorm.logpdf`.

## Considered Options

- **`NoiseModel` owns the whole `evaluate()` (subsumes the iteration harness).**
  Rejected: it relocates the shared row loop / weighting / NaN→`None` / rounding
  *into* the family, forcing the plain losses (`sos`, `sod`, …) onto a parallel
  base, blowing the byte-green budget, and breaking the M2.3 analogy (`Prior`
  never owned the proposal loop).
- **Uniform `__init__(self, config)` instead of `from_config`.** Rejected: the
  value-oracle tests construct objfuncs with positional signatures
  (`NegBinLikelihood(r, rounding)`, `ChiSquareObjective(ind_var_rounding=1)`);
  taking `config` in `__init__` would churn that byte-green file. `from_config`
  delivers ADR-0005's intent — recipe-free uniform dispatch — without it.
- **Fold `kl` into `NoiseModel` as a column-joint mode now.** Rejected: one user,
  and its real interface shape (Dirichlet-multinomial concentration vs a GP
  kernel) is unknown; per ADR-0009, harvest at the second member.
- **`lognormal` as a standalone family class.** Rejected: it would prove nothing
  about the axes. Deriving lognormal by *composing* the Gaussian family on the
  scale and location axes is what proves they are orthogonal and live.
