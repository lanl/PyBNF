# `tests/sbml_files/` — SBML / Antimony model fixtures

Standalone SBML / Antimony models used by the test suite to exercise the non-BNGL
simulation paths (the bngsim SBML/Antimony backend and, with forward sensitivities, the
gradient-fitting path of #386).

## `becker_epor.ant`

The **Becker et al. (Science 2010) EpoR core model** — BioModels `BIOMD0000000271`, in
Antimony format. An 8-reaction, 6-species deterministic ODE model of erythropoietin (Epo)
binding to its receptor (EpoR), internalisation, recycling, and degradation. No discrete
events, so it is differentiable: bngsim can carry forward output sensitivities through it,
which is what the gradient optimizers (`trf` / `lbfgs`) consume.

This is the model the **Data2Dynamics (D2D)** gradient-fitting methodology PyBNF's
gradient path is modeled on (multi-start → trust-region-reflective least squares) is
demonstrated on, in the standalone reference example
`examples/becker_d2d_gradient/` (on branch `origin/examples/becker-d2d-gradient`). It is
vendored here verbatim from that example so a **scheduler-driven** gradient smoke test
(`tests/test_gradient_optimizer.py::test_trf_multistart_smoke_through_the_scheduler_on_becker_epor`)
can exercise the Antimony → SBML forward-sensitivity gradient path end to end through
PyBNF's real scheduler — the path the small `.bngl` decay/bi-exponential recovery fixtures
do not cover.

### Provenance

`becker_epor.ant` is a copy of `examples/becker_d2d_gradient/BIOMD0000000271.ant`. The
underlying model is BioModels `BIOMD0000000271` (Becker, V. et al., *Covering a broad
dynamic range: information processing at the erythropoietin receptor*, Science
328:1404–1408, 2010). The smoke test fits the binding rates (`kon`, `koff`) against a
zero-noise synthetic time course it generates from the model's own published nominal
parameters — a self-consistent oracle, not the experimental D2D data — so the fixture is
fully self-contained and offline-reproducible.
