# Batch univariate prior families, and a three-parameter carrier with student_t record-only (issue #438 item 1)

**Status: Accepted (implemented 2026-06-26).** Closes the prior-family half of **item 1** of #438
(the low-cost Stan-parity shortlist): batch-add the univariate prior families Bayesian modelers
reach for. Builds on ADR-0010 (prior = family × scale, the registry this extends), ADR-0043 (the
new-era `parameter:` record, the labeled authoring surface), ADR-0047 (one-sided truncation, the
support floor the positive families measure against), and ADR-0022 (the log-base scale axis every
family inherits for free). The student_t **noise** family (the other half of item 1) is deferred —
see *Out of scope*.

## Why

ADR-0010 made a prior family a ~25-line `scipy.stats`-backed file plus
`@register_prior_family`, and #417 already shipped five catalog families (gamma, cauchy,
exponential, chisquare, rayleigh) on top of the original normal/uniform/laplace. So the marginal
cost of any *univariate* scipy distribution is essentially one file: it self-registers, the
registry generates its `{base}_var` / `log{base}_var` / `ln{base}_var` keywords, and both the
legacy positional grammar and the new-era `parameter:` record pick it up. #438 item 1 is the list
of distributions worth that one file each, prioritized by what modelers actually use.

This ADR adds eight families:

| family | scipy | params | support | notes |
|---|---|---|---|---|
| `half_normal` | `halfnorm(scale)` | 1 | `(0, ∞)` | weakly-informative **scale** prior (Gelman) |
| `half_cauchy` | `halfcauchy(scale)` | 1 | `(0, ∞)` | heavy-tailed scale prior |
| `beta` | `beta(α, β)` | 2 | `[0, 1]` | fractions / probabilities |
| `inv_gamma` | `invgamma(shape, scale)` | 2 | `(0, ∞)` | conjugate variance prior |
| `weibull` | `weibull_min(shape, scale)` | 2 | `(0, ∞)` | lifetime / time-to-event |
| `gumbel` | `gumbel_r(loc, scale)` | 2 | `(−∞, ∞)` | extreme-value (max) |
| `logistic` | `logistic(loc, scale)` | 2 | `(−∞, ∞)` | heavier-tailed normal sibling |
| `student_t` | `t(df, loc, scale)` | **3** | `(−∞, ∞)` | heavy-tailed **robust** prior |

Seven of the eight are pure ADR-0010 leaves. `student_t` is not — and forcing the issue's
**top-priority** family through the existing two-parameter carrier would have made it a worse
prior. That tension is the whole design content of this ADR; it had two forks, each resolved with
the issue author.

## The two decisions

### 1. student_t gets a real third parameter — the carrier extends to `p3`

A useful Student-t *prior* wants three knobs: `df` (tail heaviness — small `df` ⇒ fat tails ⇒
permissive; `df → ∞` ⇒ Normal), and `location` / `scale` to position it. This is exactly Stan's
and PyMC's `student_t(ν, μ, σ)`. But PyBNF's prior carrier was hardwired to **two** numbers:
`FreeParameter(name, type, p1, p2)`, `build_prior(keyword, p1, p2)`, `Prior.build(cls, p1, p2,
scale)`, and the parse grammar's one-/two-number branches.

The alternatives were (B) a *standardized* two-parameter `student_t` with `loc` fixed at 0, which
fits today's carrier with zero change — but a Student-t prior you cannot center is much weaker and
asymmetric to how Normal/Cauchy work; or (C) defer student_t entirely. Both sacrifice the family
the issue ranked first. **Chosen: extend the carrier to a third parameter `p3`.**

The extension is deliberately small and uniform:

- `FreeParameter.__init__` gains a **trailing** `p3=None` (so every existing positional caller —
  `set_value`'s reconstruction, the PEtab adapters, `config._load_variables` — is unchanged),
  stored as `self.p3` and threaded into `build_prior`, `set_value`, and `__eq__`.
- `build_prior(keyword, p1, p2, p3=None)` passes `p3` to the family `build`.
- Every family `build` classmethod takes a trailing-optional `p3=None`
  (`build(cls, p1, p2, scale, p3=None)`); only `StudentT.build` reads it. Putting `p3` after
  `scale` (rather than between `p2` and `scale`) keeps all the existing positional `build(...)`
  call sites — including tests calling `Uniform.build(lo, hi, LINEAR)` — working untouched. The
  one- and two-parameter families ignore `p3` exactly as the one-parameter families already
  ignore `p2`.
- The new-era record builder reads `p3 = params[2] if len(field_names) >= 3` (the loop over
  `field_names` was already arity-general; only the collapse to `p1`/`p2` needed a third slot).

Net: the carrier now carries up to three scalar parameters, and a future univariate three-knob
family is a single file with `n_params = 3` and three `field_names`.

### 2. A three-parameter family is authored only through the new-era `parameter:` record

The legacy positional grammar (`<family>_var = id p1 p2`) cannot grow a clean third number: a
three-token value **already** means a bounded-box prior plus its reflecting-bounds `b`/`u` flag
(`uniform_var = k 10 20 U`), and `config._load_variables` routes a length-3 value to exactly that.
A third *numeric* parameter would be ambiguous against that existing shape.

The new-era `parameter:` record (ADR-0043) has no such problem — every part of the line is named,
so the third value has an unambiguous home:

```
parameter: x, prior: student_t, df: 4, location: 0, scale: 2.5
```

**Chosen: `n_params >= 3` families are record-only.** `var_keyword_grammar()` omits them from the
positional partition (so `student_t_var` is not a positional keyword and a positional attempt is a
clean parse error), while they remain in `PRIOR_KEYWORD_MAP` so the record path — which resolves
`prior: student_t` → `student_t_var` → the family — works. The seven ≤2-parameter families added
here keep both surfaces, like every prior before them. This is consistent with ADR-0043's framing
of the record as the surface for richer declarations the positional grammar can't express, and it
avoids overloading the ambiguous length-3 legacy list.

A family's own `scale` field (cauchy/gamma/inv_gamma/weibull/student_t) is a distribution
parameter and is distinct from the record's `parameter_scale` transform (`linear`/`log10`/`ln`),
which is a separate key — so `prior: student_t, …, scale: 2.5, parameter_scale: log10` is
unambiguous. Every family also inherits the log/ln scale forms automatically (`logstudent_t_var`,
etc.), and the unbounded families inherit one-sided/two-sided truncation via the family-agnostic
`TruncatedPrior` (ADR-0020/0047).

## Support floors and `has_bounded_support`

The positive families (`half_normal`, `half_cauchy`, `inv_gamma`, `weibull`, and `beta`) set
`support_lo_u = 0.0`, so a one-sided truncation floors at 0 (ADR-0047), like gamma/exponential.
`beta` is the first family with a finite **both**-sided natural support (`[0, 1]`); it keeps
`has_bounded_support = False` because that flag specifically marks the *Uniform-box* semantics
(config values are the bounds, latin-hypercube participation), which beta is not — its config
values are shapes and it samples from its own density. Beta's upper support `1` is the frozen
distribution's own (a value above it already has zero density), so it needs no separate floor
attribute; truncating beta to a sub-interval of `[0, 1]` via `lower`/`upper` works through
`TruncatedPrior`. (A nonsensical truncation *above* 1 is not floor-warned — a minor, harmless gap,
since the density there is already zero.) The doubly-unbounded families (`gumbel`, `logistic`,
`student_t`) keep the default `-inf` floor.

## Out of scope

- **The student_t *noise* family** (the other bullet of item 1, a robust-regression likelihood in
  `pybnf/noise/`, ADR-0011). It needs its own decision — where the extra `df` parameter comes from
  (a fixed config value vs. another free parameter) — and is a separate, focused follow-up.
- The architectural boundaries #438 already drew stay drawn: **multivariate / joint priors**
  (`multi_normal`, LKJ, Dirichlet, Wishart) and **constrained parameter types** need a joint-prior
  abstraction PyBNF's strictly per-parameter-scalar prior (`ln_prior` sums independent scalar
  `logpdf`s) does not have. The carrier extended here is still scalar — `p3` is a third *scalar*
  knob of one univariate family, not a step toward vectors.
- `pareto` and `von_mises` from the issue's "cheap follow-ons" were dropped from this batch:
  pareto's lower support is its `scale` (parameter-dependent), which doesn't fit the constant
  `support_lo_u` floor model cleanly, and von_mises is circular (niche for a sysbio prior). Either
  is a later one-file addition if wanted.

## Consequences

- Eight new families, each oracled against its `scipy.stats` distribution (logpdf/ppf/rvs), plus
  the three-parameter carrier path oracled end-to-end through the record (`tests/test_priors.py`,
  `tests/test_config_class.py`).
- The prior carrier is now three-scalar; `build_prior` and every `Prior.build` have a uniform
  trailing-`p3` signature. No behavior change for any existing family or config.
- PEtab interop is unaffected: the importer maps a fixed set of PEtab distributions onto PyBNF
  families and the exporter refuses families PEtab v2 cannot express; the new families are
  PyBNF-native (like the existing extended catalog) and simply aren't part of the PEtab
  round-trip.
