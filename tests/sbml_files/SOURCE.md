# `tests/sbml_files/` — SBML / Antimony model fixtures

Standalone SBML / Antimony models used by the test suite to exercise the non-BNGL
simulation paths (the bngsim SBML/Antimony backend and, with forward sensitivities, the
gradient-fitting path of #386).

## `becker_epor.ant` / `becker_epor.xml`

The **Becker et al. (Science 2010) EpoR core model** — BioModels `BIOMD0000000271`, in
Antimony (`.ant`) and the equivalent SBML (`.xml`) form. An 8-reaction, 6-species
deterministic ODE model of erythropoietin (Epo) binding to its receptor (EpoR),
internalisation, recycling, and degradation. No discrete events, so it is differentiable:
bngsim can carry forward output sensitivities through it, which is what the gradient
optimizers (`trf` / `lbfgs`) consume.

This is the model the **Data2Dynamics (D2D)** gradient-fitting methodology PyBNF's gradient
path is modeled on (multi-start → trust-region-reflective least squares) is demonstrated
on, in the standalone reference example `examples/becker_d2d_gradient/` (on branch
`origin/examples/becker-d2d-gradient`). It is vendored here so the **scheduler-driven**
gradient smoke tests in
`tests/test_gradient_optimizer.py` (`test_trf_multistart_smoke_on_becker_epor_*`) can
exercise the SBML / Antimony forward-sensitivity gradient path end to end through PyBNF's
real scheduler — the path the small `.bngl` decay/bi-exponential recovery fixtures do not
cover. Two complementary cases use the two forms:

- **`becker_epor.ant`** — scored on a bare model species (`Epo_EpoR`), the dependency-free
  path (no `petab` extra), via `BngsimAntimonyModelNoTimeout`.
- **`becker_epor.xml`** — scored through a measurement-model `observableFormula`
  (`Epo_cells = Epo_EpoRi + dEpoi`, the D2D internalised-Epo observable), the ADR-0036
  observation layer (needs the `petab` math extra; a `.ant` model exposes no
  formula-observable namespace, hence the SBML form).

### Provenance

`becker_epor.ant` is a copy of `examples/becker_d2d_gradient/BIOMD0000000271.ant`;
`becker_epor.xml` is its libantimony-emitted SBML. The underlying model is BioModels
`BIOMD0000000271` (Becker, V. et al., *Covering a broad dynamic range: information
processing at the erythropoietin receptor*, Science 328:1404–1408, 2010). The smoke tests
fit the binding rates (`kon`, `koff`) against a zero-noise synthetic time course they
generate from the model's own published nominal parameters — a self-consistent oracle, not
the experimental D2D data — so the fixtures are fully self-contained and
offline-reproducible.
