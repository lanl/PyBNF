# Monine-2010 — trivalent-ligand / bivalent-receptor aggregation (PyBNF fitting job)

A PyBNF edition-2 parameter-fitting job derived from the classic multivalent-ligand aggregation
model. The **model, the fit data, and the fitted parameters** all come from Monine et al. (2010);
the trivalent ligand itself (compound 6a) was made — and its degranulation response measured — by
Posner et al. (2007):

> **[model + Fig 2a binding data + Table 1 fit]** Monine MI, Posner RG, Savage PB, Faeder JR,
> Hlavacek WS.
> **"Modeling multivalent ligand-receptor interactions with steric constraints on
> configurations of cell-surface receptor aggregates."**
> *Biophys J* 2010; **98**(1):48–56.
> PMID: [20074516](https://pubmed.ncbi.nlm.nih.gov/20074516/) ·
> DOI: [10.1016/j.bpj.2009.09.043](https://doi.org/10.1016/j.bpj.2009.09.043)
>
> **[ligand (compound 6a) + degranulation data — NOT the binding fit]** Posner RG, Geng D,
> Haymore S, Bogert J, Pecht I, Licht A, Savage PB.
> **"Trivalent antigens for degranulation of mast cells."**
> *Org Lett* 2007; **9**(18):3551–3554.
> PMID: [17691814](https://pubmed.ncbi.nlm.nih.gov/17691814/) ·
> DOI: [10.1021/ol071231z](https://doi.org/10.1021/ol071231z)

Built with the `curate-pybnf-job` skill. The job below is a **self-contained folder** — its own
model, conf, data, reproduction figure, and README with the exact adaptations, verification
results, and a ready-to-paste `_manifest.py` snippet. It is the same fitting problem as
**BioNetFit 1 example 3** (Thomas et al., *Bioinformatics* 2016, 32:798–800) and PyBNF 2019
paper corpus problem **11-TLBR** (Mitra et al., *iScience* 2019, 19:1012–1036), re-expressed on
the edition-2 surface.

## The biochemistry

A **trivalent ligand** crosslinks **bivalent cell-surface receptors** into aggregates. Because
each ligand has three arms and each receptor two sites, crosslinking builds complexes that grow
**without bound** — the reaction network is not finite — so the model is simulated **network-free
with NFsim**. Three parameters are fit against the bound-ligand fraction FL measured across a
titration of total ligand: an overall readout scale `alpha`, the free-ligand/receptor-site
equilibrium constant `K1`, and the crosslinking equilibrium constant `K2`. Dissociation (`koff`)
and the cell geometry (Table 1 of Monine 2010) are fixed.

## The job

| slug | fits | simulator | flavor | status |
|---|---|---|---|---|
| [`tlbr`](tlbr/) | bound-ligand fraction FL vs. total-ligand dose-response, 12 doses (3 params) | **NF** (NFsim, network-free) | quantitative, **PEtab-exportable**, `sos` | ✅ tier-1 + PEtab round-trip + reproduction (median 2.4 % rel err) · **validated** (`tlbr/VALIDATION.md`, 91/100) |

Quantitative and **PEtab-v2-exportable** (unlike native-only BPSL jobs). Network-free but
**not heavy** — a single 12-dose scan runs
in ~11 s, so the reproduction runs on a workstation in ~1 minute; a full metaheuristic fit is
still best on a cluster (the published fit used a large population there). A linted PEtab v2
bundle can be generated on demand with `tlbr/make_petab.py`; generated files are not committed
in this gallery.

## Source materials

- **Primary paper (model, Fig 2a binding data, Table 1 fit, SI):** Monine 2010 *Biophys J* 98:48–56
  (+ Supporting Material). Validated against these — see [`tlbr/VALIDATION.md`](tlbr/VALIDATION.md).
- **Model / classic conf lineage:** PyBNF `examples/real-world/Monine-2010/tlbr/` and `examples/tlbr/`
  (BioNetFit-1-era) — model blocks identical to the paper's TLBR model.
- **Best-fit used for the reproduction:** **Monine 2010 Table 1 (TLBR): α=0.816, K1=0.467,
  K2=87.03** — the paper's *own* reported result. (A separate PyBNF/Mitra-2019 re-fit, RuleHub
  `11-TLBR/fit_ss` init7 α=0.746/K1=0.109/K2=33.6, is a different point of the same sloppy K1–K2
  valley; it minimizes the FL-space `sos` but not Monine's λ-space RMS, and is **not** used here.)

## Run

```bash
export BNGPATH="$HOME/Simulations/BioNetGen-2.9.3"   # folder with BNG2.pl
cd examples/real-world/Monine-2010/tlbr
pybnf -c tlbr.conf
# reproduction figure (NFsim at Monine's Table-1 fit vs. Fig 2a):
python make_reproduction.py
```
