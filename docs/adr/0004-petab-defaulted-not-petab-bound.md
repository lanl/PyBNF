# Prior and noise-model catalogs are PEtab-defaulted, not PEtab-bound

PEtab v2 assumes a model's deterministic prediction is the conditional **median**
of the observation model. That is only correct for noise that is symmetric on the
prediction's own scale; for noise additive on a different scale (e.g. lognormal,
where the prediction is the median but the mean is `prediction·exp(σ²/2)`), the
appropriate location depends on what the model output physically represents. We
decided to model **noise models as three orthogonal axes — distribution family ×
scale-the-noise-is-additive-on × location interpretation (mean/median/mode)** —
and **priors as distribution family × scale** — taking PEtab's conventions as
*defaults* for interoperability, but making location and scale first-class,
overridable axes. Non-probabilistic losses (`sos`, `sod`) remain plain objectives,
not noise models. This buys full PEtab catalog parity and clean interop without
inheriting PEtab's median bias, and the orthogonal axes avoid fusing
distribution+scale into combinatorial magic strings (`lognormal_var`,
`loglaplace_var`, …).

## Consequences

- A future PEtab-problem importer maps onto these objects as the natural "two-adapter" validation that the abstraction is right (native `.conf` and a PEtab problem → the same internal `Prior`/`NoiseModel` objects).
- `objective.py`'s probabilistic members (`chi_sq`, `neg_bin`) become `NoiseModel`-derived; the file may become a package as the catalog grows.
