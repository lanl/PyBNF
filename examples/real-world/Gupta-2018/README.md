# Gupta-2018 — FcεRI stochastic parameter recovery

The [`fceri_gamma/`](fceri_gamma/) job is the stochastic FcεRI γ-chain signaling
benchmark of Gupta and Mendes (2018), included in the PyBioNetFit real-world corpus of
Mitra et al. (2019). It fits 20 kinetic parameters to synthetic time-course data with
exact Gillespie SSA and replicate smoothing.

The model expands to roughly 58,000 reactions, so it is retained as a cluster-scale
reference. Default CI validates its edition-2 configuration; the compact
[`../Rijal-2025/`](../Rijal-2025/) jobs provide workstation-scale executable SSA coverage.

Run from the repository root with BioNetGen configured:

```bash
pybnf -c examples/real-world/Gupta-2018/fceri_gamma/fceri_gamma.conf
```
