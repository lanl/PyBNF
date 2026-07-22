# Salazar-Cavazos-2019 — multisite EGFR phosphorylation, adaptor abundances & dimer lifetimes (PyBNF fitting job)

A PyBNF edition-2 parameter-fitting job derived from a study that combined multiplex single-molecule
pull-down (SiMPull) measurements with rule-based modeling to dissect how EGFR is phosphorylated at
multiple C-tail tyrosines. The **model, the fit data, and the fitted parameters** all come from the
same paper (the authors, several of whom are PyBNF/BioNetGen developers, shipped their complete
PyBioNetFit setup as a supplement):

> Salazar-Cavazos E, Franco Nitta C, Mitra ED, Wilson BS, Lidke KA, Hlavacek WS, Lidke DS.
> **"Multisite EGFR phosphorylation is regulated by adaptor protein abundances and dimer
> lifetimes."**
> *Mol Biol Cell* 2020; **31**(7):695–708.
> PMCID: [PMC7202077](https://pmc.ncbi.nlm.nih.gov/articles/PMC7202077/) ·
> DOI: [10.1091/mbc.E19-09-0548](https://doi.org/10.1091/mbc.E19-09-0548)

Built with the `curate-pybnf-job` skill. The job below is a **self-contained folder** — its own
model, conf, data, reproduction figure, and README with the exact adaptations, verification results,
and a ready-to-paste `_manifest.py` snippet. Because the authors published their actual BioNetFit
model + data + config (supplement `mc-e19-09-0548-s03/PyBNF-fitting-setup/`), the fit recipe is
**fully known**, not reconstructed — the same gold-standard situation as `Erickson-2019/igf1r`.

## The biochemistry

EGF reversibly binds EGFR; two liganded receptors **dimerize** (modeled as a receptor state change,
one kinase becoming the activator and the other the receiver); the activated dimer
*trans*-phosphorylates three C-tail tyrosine classes — **Y1068**, **Y1173**, and a lumped **"other"
YN**; constitutive phosphatases dephosphorylate; the adaptors **Grb2** (at pY1068) and **Shc1** (at
pY1173) bind. The paper's finding — reproduced by the fit — is that the *level* of phosphorylation
tracks **adaptor protein abundances** and receptor **dimer lifetimes**. The network is finite
(dimerization is a state change, not an unbounded aggregate; 75 species), so it is simulated as a
deterministic ODE. The six fitted parameters are the two adaptor copy numbers and four
phospho-turnover constants.

## The job

| slug | fits | simulator | flavor | status |
|---|---|---|---|---|
| [`egfr_simpull`](egfr_simpull/) | SiMPull % phosphorylation (pY1068, pY1173, pan-PY): EGF dose-response (3 doses) **+** 25 nM time course, jointly (6 params) | **ODE** | quantitative, **PEtab-exportable**, `chi_sq` | ✅ tier-1 + PEtab round-trip + reproduction (median 10.4 % rel err) · **validated** ([`egfr_simpull/VALIDATION.md`](egfr_simpull/VALIDATION.md), 93/100) |

Quantitative and **PEtab-v2-exportable** (unlike native-only BPSL jobs). **Not heavy** —
the 75-species network generates in ~1 s and a full fit runs on a
workstation. An independent PyBNF fit reaches the same objective as the published best-fit
(chi_sq 361 vs. 360), confirming the reported parameters are the global optimum. A linted PEtab
v2 bundle can be generated on demand with `egfr_simpull/make_petab.py`; generated files are not
committed in this gallery.

## Source materials

- **Primary paper (model, data, fit, SI):** Salazar-Cavazos 2020 *Mol Biol Cell* 31:695–708
  (PMC7202077). Validated against the authors' own supplement files — see
  [`egfr_simpull/VALIDATION.md`](egfr_simpull/VALIDATION.md).
- **Authors' PyBioNetFit setup:** `mc-e19-09-0548-s03/PyBNF-fitting-setup/` —
  `190127_CHO_EGFR_forBNF.bngl` (fitting model), `dose_resp.exp` + `EGF_25nM.exp` (SiMPull data),
  `fit_v1_28.conf` (de/chi_sq fit), plus `fit_bootstrap.conf` / `fit_pt.conf` (uncertainty analyses).
- **Best-fit used for the reproduction:** the six values in `190127_CHO_EGFR_best-fit.bngl` — the
  authors' own reported PyBNF fit. Our `.bngl` generates a network **byte-identical** to that model.

Not yet built (optional future slugs from the same paper): the cell-line variants
(`190127_HeLa/HMEC/MCF10A.bngl`, same model with cell-specific copy numbers), the Epigen-ligand and
L858R-mutant fits, and the parallel-tempering UQ (`fit_pt.conf`).

## Run

```bash
export BNGPATH="$HOME/Simulations/BioNetGen-2.9.3"   # folder with BNG2.pl
cd examples/real-world/Salazar-Cavazos-2019/egfr_simpull
pybnf -c egfr_simpull.conf
# reproduction figure (model at the authors' best-fit vs. the SiMPull data):
python make_reproduction.py
```
