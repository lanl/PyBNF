# PEtab prior catalog parity: the five missing families land via the family registry, with a one-parameter grammar form (issue #417)

**Status: Accepted and implemented (decision + implementation 2026-06-20).** The five PEtab v2
prior families PyBNF lacked -- `cauchy`, `gamma`, `exponential`, `chisquare`, `rayleigh` -- are
now registered `Prior` families (`pybnf/priors/<family>.py`, `@register_prior_family`), each one
file as ADR-0010 promised. The var grammar admits a one-number form for the one-parameter
families, `config._load_variables` builds them, and `parameters.py` maps them bidirectionally.
Oracles green: `tests/test_priors.py::TestCatalogFamilies` (scipy logpdf/ppf/rvs value oracles +
keyword-map resolution), `tests/test_petab_import.py::TestImportExtensionsRoundTrip`
(native `.conf` -> PEtab -> `.conf` byte-equal round trip per family + the import-direction
truncation unit oracle), and the preserved one-sided-truncation / unknown-distribution boundary
raises. A `FrozenPrior` base (`priors/base.py`) now holds the scipy-delegating
`logpdf`/`rvs`/`ppf`/`support` shared by every location/scale/shape family (Normal and Laplace
migrated onto it; Uniform keeps its custom latin-hypercube `ppf`, NoPrior its null object).

ADR-0010 made a prior family a single registration; ADR-0019/0023/0032 made the PEtab
parameters mapper bidirectional but left five families as a documented `NotImplementedError`
gap. This closes it.

## The parameterizations are verified against petab, not guessed

The load-bearing risk here is a wrong parameterization (a gamma read as shape+rate instead of
shape+scale silently fits the wrong prior). Each mapping is verified against petab's **own**
`v1.distributions` classes (`Cauchy`/`Gamma`/`Exponential`/`ChiSquare`/`Rayleigh`), not a guess:

| PEtab `priorDistribution` | priorParameters | scipy | params |
|---|---|---|---|
| `cauchy` | `(loc, scale)` | `cauchy(loc, scale)` | 2 |
| `gamma` | `(shape, scale)` | `gamma(a=shape, scale=scale)` | 2 |
| `exponential` | `(scale)` | `expon(scale=scale)` | 1 |
| `chisquare` | `(dof)` | `chi2(df=dof)` | 1 |
| `rayleigh` | `(scale)` | `rayleigh(scale=scale)` | 1 |

Two spots a guess would get wrong: **gamma is shape + scale** (not shape + rate), and
**exponential's parameter is the scale `1/rate`** (not the rate). PEtab defines **no `log-` form**
for these five, so only the linear keyword maps; their native `log{stem}_var` keywords still
exist (the registry generates `{stem}_var` + `log{stem}_var` for every family) but have no PEtab
`priorDistribution`, so the exporter refuses them -- the native surface is a superset of v2.

## The one-parameter grammar form

The var grammar assumed two numbers (`<keyword> = <p> <num> <num>`); `exponential` (scale),
`chisquare` (dof), and `rayleigh` (scale) are **one-parameter**. The second number is now
optional (`num - Optional(num) - Optional(flag)`); the trailing reflecting-bounds flag is letters
and a number is digits, so there is no ambiguity. `config._load_variables` reads a one-number
declaration as `(p1,)` and builds the `FreeParameter` with `p2 = None`; each one-parameter
family's `build(p1, p2, scale)` reads `p1` and ignores `p2`. The importer (`_free_parameters`)
and exporter (`_free_parameters_from_conf`) both emit/read the one-number line. The two-parameter
families (cauchy `(loc, scale)`, gamma `(shape, scale)`) reuse the existing two-number shape.

## Truncation: two-sided works; one-sided stays deferred (#417), at parity with normal/laplace

PEtab bounds truncate a prior. `_truncation_box` now measures bounds against each family's
**natural lower support** (`_FAMILY_META`: 0 for the half-bounded gamma/exponential/chisquare/
rayleigh, -inf for the doubly-unbounded normal/laplace/cauchy, and 0 for any log form). A real
PEtab catalog prior carries **finite two-sided bounds** (PEtab requires bounds on estimated
parameters), so it imports as a **two-sided** `TruncatedPrior` (ADR-0020) -- already supported.
A *one-sided* truncation (one bound infinite) still raises `NotImplementedError`: the
triangle-wave reflection fold (`FreeParameter._reflect`) needs two finite bounds. This is exactly
the boundary normal/laplace already have, so the catalog lands at **full parity** with them, not
in a half-built state; extending one-sided truncation to every unbounded-support family is the
genuinely separate **#417** cross-cutting change (a single-wall reflection + a half-open
`TruncatedPrior` + the `_initialization_bounds_u` fallback), deferred.

## A consequence worth recording: an unbounded native prior is not valid PEtab

PEtab requires finite `lowerBound`/`upperBound` on every estimated parameter, but PyBNF's
native surface has **no truncation grammar** for unbounded-support families (a `gamma_var = k 2 3`
is unbounded). So such a prior exports to PEtab with **blank bounds**, which `petablint` rejects --
a **pre-existing** limitation shared by `normal_var`/`laplace_var`, not introduced here. The
native byte-equal round trip is therefore the internal consistency oracle for the catalog
(reader inverts writer); the import direction (a bounded PEtab prior -> a truncated
`FreeParameter`) is the petab-shaped oracle, tested at the unit level. A native truncation
grammar (so an unbounded prior can carry bounds and export valid PEtab) is the natural companion
to #417 and is filed there.

See ADR-0010 (the family registry this realizes), 0020 (the two-sided truncated decorator), 0003
(scale-agnostic priors, no Jacobian), 0019/0023/0032 (the bidirectional PEtab parameters mapper).
Closes the catalog-parity checkbox on #407; #417 (one-sided truncation + native truncation
grammar) remains open.
