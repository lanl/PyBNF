# Prior is a scale-agnostic family in u-space; FreeParameter owns the scale (M2.3 build shape)

`FreeParameter` mixed three concerns behind the four legacy `*_var` type strings:
the **distribution family**, the **parameter scale**, and the (scale-aware)
**proposal arithmetic**. M2.3 extracts the prior into a `pybnf/priors/` package as
a **behavior-preserving** change — the contract is the existing green
`tests/test_distributions.py` + `tests/test_freeparameter_class.py`, kept
byte-green (extend, don't weaken). Building on ADR-0003 (prior evaluated in the
parameter's own scale, **no** change-of-variables Jacobian), we settled this shape:

- **`Prior` is a pure distribution family living entirely in the sampling space
  `u`** — one file per family under `pybnf/priors/` (`Normal`, `Uniform`, …). It
  does **not** know about scale. **`FreeParameter` owns a first-class `Scale`
  object** (`Linear` / `Log10`) that applies `θ↔u` (`u = log10 θ`), shared with the
  proposal arithmetic, so the `log10`/`10**` boundary lives in exactly **one**
  place: `prior_logpdf(θ) = prior.logpdf(scale.forward(θ))`, `sample_value() =
  scale.inverse(prior.rvs())`.

- **`NoPrior` is a first-class null-object** for `var`/`logvar` (Simplex start
  points): `logpdf → 0`, `rvs`/`ppf` raise, `has_prior = False`. It still carries a
  scale (`logvar = (NoPrior, Log10)`) — which is *why* scale can't belong to the
  family object.

- **Two boxes, deliberately separated.** *Support* (the family's nonzero region —
  finite for `Uniform`, infinite for `Normal`/`Laplace`) lives on the **family**.
  *Reflecting Bounds* (the proposal fold box) live on **`FreeParameter`** as
  `bounded = (b/u flag) AND family.has_bounded_support`. They decouple: an unbounded
  `uniform_var` has a finite support yet no reflecting bounds.

- **A prior-family registry** (`register_prior_family` in the top-level
  `registry.py`, mirroring `register_fit_type`/`register_objfunc`, ADR-0005) is the
  single source of truth. The regular keyword naming `{base}_var` / `log{base}_var`
  lets a registered family generate both its linear and log10 forms; `parse.py`'s
  grammar (the `b_var_def_keys` vs `var_def_keys` partition **is**
  `has_bounded_support`) and `config._load_variables`'s `{keyword: (family, scale)}`
  map both **derive** from the registry. Adding a family (e.g. `Laplace`) is one
  registration, grammar included.

- **`FreeParameter`'s constructor and public surface stay frozen.** It still takes
  the legacy type string and resolves `(family, scale)` **internally** from the
  registry-derived keyword map; `.type`, `.log_space`, `.bounded`,
  `.lower/upper_bound`, `.default_value`, `._distribution` (now a passthrough
  property), `.prior_logpdf`, `.sample_value`, `.add/.set_value/.diff` are all
  preserved. After M2.3 **no algorithm switches on `var.type`** (LH →
  `has_bounded_support` + `value_from_quantile`; the `ln_prior` box-warning →
  `has_bounded_support`; `adaptive_mcmc`'s `'log' in type` → `log_space`).

## Considered Options

- **`Prior` holds the scale and evaluates the density in `θ`.** Rejected: it
  duplicates the `θ↔u` transform across the prior *and* the proposal arithmetic
  (the drift/bug class ADR-0003 warns against), and it cannot represent a
  no-prior-but-log-scale `logvar`.
- **Plain dict for the `*_var`→(family, scale) map instead of a registry.**
  Rejected for the same reason ADR-0005 gives: a registry lets each family
  self-register *in its own file* (the "one file per family" goal), rather than a
  central dict that must be hand-edited for every new family.
