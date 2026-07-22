# Mitra-2019 — PyBioNetFit receptor examples

These ligand/receptor binding and phosphorylation jobs come from the real-world example
suite accompanying Mitra et al. (2019), *PyBioNetFit and the Biological Property
Specification Language*.

- [`receptor/`](receptor/) is the deterministic ODE fit with ligand-free
  pre-equilibration followed by ligand stimulation.
- [`receptor_nf/`](receptor_nf/) is its network-free NFsim sibling, using a fixed-time
  pre-equilibration because NFsim does not provide a steady-state solve.

Both are edition-2 ports of the classic jobs under `examples/`; the model, data, and fit
remain the same while the experiment protocol is synthesized from the configuration.

Run from the repository root with BioNetGen configured:

```bash
pybnf -c examples/real-world/Mitra-2019/receptor/receptor.conf
```
