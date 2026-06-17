# Initialization distribution is separate from the objective prior (issue #413)

Status: Accepted.

PyBNF historically used one `Prior` object for two jobs:

- objective prior density, via `FreeParameter.prior_logpdf`;
- start-point generation, via `FreeParameter.sample_value` and the Latin-hypercube path.

Those are different concepts. A tight Bayesian prior can be a good regularizer
while being a poor chain initializer, because convergence diagnostics such as
R-hat are more informative when chains start over-dispersed. PEtab v2 also treats
the prior as objective-only and initializes from parameter bounds.

This ADR adds an explicit initialization distribution on `FreeParameter`:

- `prior` is the default and preserves native `.conf` behavior exactly.
- `bounds` draws uniformly over a finite initialization box in PyBNF's sampling
  space `u` (`theta` for linear parameters, `log10(theta)` for log parameters).

The algorithm layer now asks each parameter for `sample_initial_value` and
`initial_value_from_quantile` instead of using prior sampling directly. The old
`sample_value` method remains prior sampling, so prior-family tests and external
callers keep their previous meaning.

The PEtab importer sets `bounds` initialization when a row has a finite two-sided
parameter box, storing those row-level bounds separately from the objective prior
support. This is the important decoupling: importing a tight normal objective
prior no longer also concentrates the initial search points near that normal's
mean.

Native config gains a global `initialization_distribution = prior|bounds` key.
The existing `initialization = lh|rand` key still chooses the sampling scheme;
the new key chooses the per-coordinate distribution used by that scheme.

Relevant ADRs: 0003 (prior/proposal in sampling space), 0010 (prior family vs
FreeParameter scale split), 0020 (truncated priors and finite reflecting boxes).
