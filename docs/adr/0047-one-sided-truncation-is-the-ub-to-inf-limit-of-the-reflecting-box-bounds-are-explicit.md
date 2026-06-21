# One-sided truncation is the `ub → ∞` limit of the reflecting box; bounds are explicit `±inf` with scale-aware support floors, never specified by absence (issue #432)

**Status: Accepted → implemented (2026-06-21).** Amends **ADR-0020**,
which gave the unbounded-support families (`normal`/`laplace`/`cauchy` + log forms) a
*two-sided* truncated prior (a finite reflecting box) and explicitly carved out
*one-sided* truncation — a single finite bound with the other infinite — as a `raise`
across both adapters (native `parameter:` record, PEtab importer). This ADR removes that
carve-out: it pins (a) the **native grammar** for an open side, (b) the **half-bounded
containment scheme**, and (c) the two **design principles** that fell out of working the
problem against PEtab v2. Built across the seven sites in the wiring section below
(including two beyond the issue's four: the `parse.py` `num` token and the importer's
conf-writer — a half-bounded import must reach a *runnable* conf, which also closed a
latent two-sided gap). Covered by half-bounded unit + round-trip tests in
`test_priors` / `test_freeparameter_class` / `test_config_class` / `test_petab_parameters`
/ `test_petab_import`; full fast-tier suite green (2115 passed). Issue #432.

## What ADR-0020 left raising

ADR-0020's reflecting box needs *two* finite walls (a finite width to fold into), so three
sites raise on one-sided truncation, consistently and non-silently:

- `config._free_parameter_from_record` — `bounds need both 'lower' and 'upper' (one-sided
  truncation is not supported)` (`pybnf/config.py:1631-1633`).
- `FreeParameter.__init__` — truncating an unbounded-support prior `needs both lb and ub …
  one-sided truncation has no finite reflecting box` (`pybnf/pset.py:1748-1758`).
- `petab.parameters._truncation_box` — `NotImplementedError` on a PEtab bound that
  truncates one side while the other is infinite (`pybnf/petab/parameters.py:302-310`).

The capability — not the boundary — is what was missing. This ADR adds it.

## The grammar: an open side is an explicit infinity, never an absence

A `parameter:` record's `lower`/`upper` are a **pair**: either both present or both omitted.
We do *not* let a missing `upper` mean "+∞" (specification by absence). An unbounded side is
spelled with an explicit infinity at the **family's natural support endpoint in θ**:

- **Doubly-unbounded family** (`normal`/`laplace`/`cauchy`, linear scale) — floor `-inf`,
  ceiling `inf`. Open-below is `lower: -inf`.
- **Positive-support family** (`gamma`/`exponential`/`chisquare`/`rayleigh`, linear scale;
  and *any* `log*`/`ln*` form, whose θ-support is `(0, ∞)`) — floor `0`, ceiling `inf`.
  Open-below is `lower: 0` (on a log scale, `Scale.forward(0) = log10(0) = -∞`, so a `0`
  θ-floor is exactly the open-below box in sampling space `u`).

These are the **prior** families — distributions over a *fit parameter* (`pybnf/priors/`,
`PRIOR_KEYWORD_MAP`). A *likelihood/noise* family is a different layer — a distribution over
the *observed data* given the prediction (`objective.py`) — so `neg_bin` and the per-point
Gaussian/Laplace noise models are deliberately **absent** here despite `neg_bin`'s positive
support. The one positive quantity a likelihood introduces — `neg_bin`'s dispersion `r`,
estimated as `r__FREE` (`objective.py:682-684`) or fixed as `neg_bin_r` — is, when estimated,
an *ordinary free parameter* whose `r > 0` rides on whatever prior family/scale it is given (a
`gamma`/`exponential`/log-scale prior, all above). Principle 1 covers it; nothing here
special-cases the noise layer.

Two consequences make this a clean superset of ADR-0020 rather than a break:

- **Omit-both stays the untruncated shorthand.** Leaving out *both* `lower` and `upper` is
  still the symmetric untruncated prior (ADR-0020's existing default). `lower: <floor>,
  upper: <ceiling>` is its explicit spelling; the two are equivalent. The pairing rule
  (`(lower is None) != (upper is None)` → raise) is *kept* — you cannot say half a thing by
  leaving one side blank, only by writing its infinity.
- **`±inf` is already the internal representation.** `FreeParameter` stores an untruncated
  prior as `lower_bound = -inf, upper_bound = inf` (`pset.py:1740-1741`); the half-bounded
  case is `lower_bound = 0, upper_bound = inf`. The surface infinity maps straight onto the
  value the constructor already holds — no absence↔∞ translation at the boundary, and the
  PEtab round-trip becomes a value pass-through (PEtab encodes an open side the same way; see
  the principles below).

### The graded sentinel/floor rule (a sloppy open side is forgiven; a finite contradiction is not)

For a **positive-support** family the lower side has a natural θ-floor of `0`. A bound at or
below that floor truncates nothing (the density is already zero there), so the action is
graded by how likely the input is a *confused config* rather than by its numerical effect:

| `lower` on a positive-support family | meaning | action |
|---|---|---|
| `0` | canonical "open below to the floor" | **silent** |
| `-inf` | "no lower wall" — right intent, wrong-scale sentinel | **warn + canonicalize to `0`** |
| finite, `> 0` | a real truncation wall above the floor | accept |
| finite, `< 0` | a finite wall inside the zero-density region | **error** |

The `-inf` case is lossless and unambiguous ("no wall below" ≡ "down to the support floor"),
so it earns a forgiving warn-and-fix that names the parameter and points at the canonical
`0`. A *finite* sub-floor bound is almost never intentional — it is a tell for a real mistake
(wrong family, e.g. `normal` where `gamma` was meant; wrong scale; a typo) — and silently
clamping a deliberately-typed number would hide it, so it raises, consistent with the
fail-loud pattern the three sites above already use. This subsumes and relaxes the current
strict log guard (`config.py:1639`, which raises on any `lower <= 0`): `0`/`-inf` become
legal "open below," finite negatives still raise. The upper side needs no such rule here —
the four positive-support families are all `[0, ∞)`, unbounded above, so `upper: inf` is
always valid. Uniform priors continue to reject any infinity (a uniform needs two finite
walls — its support *is* its box).

## The containment scheme: reflect at the single wall (the `ub → ∞` limit of the fold)

The two-sided fold `FreeParameter._reflect` confines a proposal to `[lb, ub]` with a periodic
triangle wave of period `2(ub − lb)`, mirroring at both walls. **As `ub → ∞` the period → ∞
and the upper fold recedes to infinity; what remains is a single reflection at the finite
wall** — `x' < lb ↦ 2·lb − x'` (or the mirror image for an open lower side). So the
half-bounded fold is **not a new scheme** — it is the limiting case of the operator already in
the code. Single-wall reflection of a symmetric proposal is itself symmetric (a
measure-preserving involution about the wall), so plain Metropolis still targets the truncated
posterior — exactly the property ADR-0020 relied on for the two-sided box, and ADR-0003's
"target defined in `u`."

**Sampling needs no fold at all.** The `TruncatedPrior` decorator's inverse-CDF
(`rvs`/`ppf`) generalizes directly: `Z = inner.cdf(hi_u) − inner.cdf(lo_u)` evaluates the
open side at `cdf(±∞) ∈ {0, 1}` (a lower wall gives `Z = 1 − inner.cdf(lo_u)`; an upper wall
`Z = inner.cdf(hi_u)`), and a draw `inner.ppf(inner.cdf(lo_u) + q·Z)` with `q ∈ [0, 1)` lands
in the half-line by construction and is finite almost surely (`q` is never exactly `1`, so
`ppf` never returns `inf`). `logpdf` is unchanged: `inner.logpdf(u) − log Z` inside the
half-line, `−inf` outside. `Z` is still parameter-independent (the wall is fixed), so it
cancels for inference and is kept only so `prior_logpdf` reports the true normalized density
(ADR-0020's rationale, unchanged).

This is the one genuinely-new design call, and reflection wins over the alternative
(reject-and-resample, below) precisely because one-sided truncations put mass **at the finite
wall** — a positive rate constant with an exponential prior, or a `normal` truncated at `0`,
peaks near the boundary — which is exactly where reject-and-resample mixes worst.

## Two principles this surfaced (and PEtab v2 already lives by)

Working the inconsistency "what if `lower: -inf` but the parameter must be positive (a rate
constant)?" against PEtab v2 produced two principles worth recording as **non-goals/contracts**,
because PEtab v2 deliberately made the same calls:

1. **No physical-positivity guard. Admissibility is carried by the scale/family/bound, not a
   separate flag.** A BNGL parameter is just a symbol substituted into the model; nothing
   declares "this is `k_on`, must be > 0." PyBNF therefore *cannot* validate prior-support
   against physical admissibility, and will not pretend to. The modeler declares positivity
   through the idioms that already exist — a **log scale** (`space: log10`, θ = 10ᵘ, positive
   by construction; on which `lower: -inf` cannot even arise), a **positive-support family**,
   or `lower: 0`. A `normal` linear prior with `lower: -inf` on a rate constant is a *modeler*
   error PyBNF can't see at parse time; its runtime symptom is a negative proposal → a failed
   or non-physical simulation → a bad objective → the sampler moving away (caught, not silently
   wrong). This mirrors PEtab v2 **removing the `parameterScale` column** — done because
   layering scale/positivity semantics on top of priors was *"a constant source of confusion
   and … not well-defined"* — and we will not re-introduce that ambiguity with a guard PyBNF
   cannot honestly enforce.

2. **A deliberate native↔PEtab divergence on mandatory bounds.** PEtab v2 *requires*
   `lowerBound`/`upperBound` for every estimated parameter ("required if estimate=true, for
   optimization ranges") and truncates the prior to them; an untruncated prior is spelled with
   the natural-domain endpoints (the spec's own example: *"`0` and `inf` for log-normal"*).
   PyBNF native keeps **omit-both → untruncated** (principle above and ADR-0020), because
   PyBNF's worldview is *prior-primary* — a `normal` prior is a complete, proper distribution
   with no box, and pure-MCMC fits routinely want an unbounded prior — whereas PEtab's is
   *optimizer-primary* (it needs a box). Both surfaces still land on the **same**
   `FreeParameter`: the native shorthand resolves internally to the same `-inf/inf` (or
   `0/inf`) the PEtab importer reads explicitly. The two-adapter proof (ADR-0004) holds; the
   surfaces differ only in whether the bound is mandatory.

## Wiring (the sites it touched)

The four sites the issue named, plus two the end-to-end path required (the native grammar
token and the importer's conf-writer) — found because a half-bounded import must reach a
*runnable* conf, not just an in-memory `FreeParameter`:

- **Core (`pset.py`)** — `FreeParameter.__init__` accepts one infinite `u`-wall (both infinite
  ≡ untruncated; both finite ≡ the two-sided box). Open-ness is detected in sampling space `u`
  via a warning-safe `_bound_to_u` (a log family's floor `theta 0`/`-inf` → `u = -inf` without
  tripping `log10(0)`). `_reflect` gains single-wall branches (mirror at the lone finite wall);
  `TruncatedPrior` already handles one infinite `u`-bound (`Z` via `cdf(±∞)`), so it was left
  unchanged.
- **Prior families (`priors/*.py`)** — a `support_lo_u` class attribute (the family's natural
  `u`-floor: `-inf` by default, `0` for gamma/exponential/chisquare/rayleigh) so the floor is
  derived as `scale.inverse(support_lo_u)`, not a hardcoded list.
- **Native grammar (`parse.py`)** — the `num` token accepts a signed `inf` (tried before the
  real-number branch so `-inf` matches whole), so an open side is a parseable conf token.
- **Native surface (`config.py`)** — drop the both-required raise; keep the pairing rule (new
  message), apply the graded sentinel/floor rule, and require finite bounds for a uniform box.
- **PEtab importer (`parameters.py`)** — `_truncation_box` replaces its `NotImplementedError`
  with the half-bounded mapping (the covered side → the support endpoint, the other → a finite
  wall).
- **PEtab importer conf-writer (`import_.py`)** — a truncated free parameter (two-sided *or*
  half-bounded) is emitted as a new-era `parameter:` record (the only grammar carrying bounds),
  not the legacy `*_var = name p1 p2` line, which silently dropped the box — also closing a
  latent two-sided gap.
- **PEtab exporter (`parameters.py`)** — a half-bounded native prior already emits a valid
  PEtab row (the finite wall on the truncated side, an explicit `±inf` on the open side) via
  `trunc_lb`/`trunc_ub` and the `inf`-aware `num` serializer — no code change, only docstrings.

## Considered Options

- **Reject-and-resample on the half-line (rejected).** Doing nothing special already
  "works": `TruncatedPrior.logpdf` is `−inf` outside the box, so an out-of-bounds proposal is
  auto-rejected by the Metropolis ratio — zero new proposal code. Rejected anyway because a
  one-sided truncation concentrates mass *at* the finite wall, which is exactly where
  reject-and-resample's acceptance collapses; reflection has no such penalty and is the
  consistent extension of ADR-0020's choice.

- **Specification by absence — a missing `upper` means `+∞` (rejected).** Concise, but a
  record with `lower` present and `upper` absent is ambiguous to a reader (intended open side,
  or a forgotten field?). Explicit `±inf` is unmistakable, keeps the pairing invariant intact,
  maps straight onto the internal representation, and matches how PEtab encodes an open side.

- **A "this parameter is positive" annotation / physical-positivity guard (rejected).** This
  is the very thing PEtab v2 deleted with `parameterScale`. PyBNF has no machine-readable
  positivity to check against, so the guard would be unverifiable and re-introduce the
  confusion PEtab removed. Positivity is declared through scale/family/bound; see Principle 1.

- **Edit ADR-0020 in place (rejected).** ADRs are immutable decision records. This adds a new
  one that amends, and tags the affected bullet in ADR-0020 with a pointer here.

Relevant ADRs: **0020** (the two-sided reflecting box this extends — its "two-sided only"
bullet is amended), **0003** (target defined in `u`, where the `Z`-cancels-for-inference and
symmetric-proposal reasoning lives), **0010** (Prior-family-vs-`FreeParameter` Support/
Reflecting-Bounds seam), **0043** (the `parameter:` record where `lower`/`upper` live),
**0022** (the explicit log10/ln scale, on which the positive-support floor depends), **0004**
(the two-adapter proof the native↔PEtab divergence is measured against). Issues: **#432**
(this tracker), **#411** (`TruncatedPrior`/box), **#417** (the native two-sided grammar this
opens up), **#407** (the PEtab adapter).
