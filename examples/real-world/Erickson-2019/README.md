# Erickson-2019 — IGF1 / IGF1R receptor binding (PyBNF fitting job)

A PyBNF parameter-fitting job reproducing the IGF1-IGF1R binding parameterization of Erickson et
al. (2019). The **fitting problem** — estimate the harmonic-oscillator rate constants from
radioligand competition and dissociation data — and its data both come from Kiselyov et al. (2009):

> **[fitting problem / job source]** Erickson KE, Rukhlenko OS, Shahinuzzaman M, Slavkova KP,
> Kholodenko BN, Hlavacek WS, et al. **"Modeling cell line-specific recruitment of signaling
> proteins to the insulin-like growth factor 1 receptor."**
> *PLoS Comput Biol* 2019; **15**(1):e1006706.
> PMCID: [PMC6353226](https://pmc.ncbi.nlm.nih.gov/articles/PMC6353226/) ·
> DOI: [10.1371/journal.pcbi.1006706](https://doi.org/10.1371/journal.pcbi.1006706)
>
> **[model + data origin]** Kiselyov VV, Versteyhe S, Gauguin L, De Meyts P.
> **"Harmonic oscillator model of the insulin and IGF1 receptors' allosteric binding and
> activation."** *Mol Syst Biol* 2009; **5**:243.
> PMID: [19225456](https://pubmed.ncbi.nlm.nih.gov/19225456/) ·
> DOI: [10.1038/msb.2008.78](https://doi.org/10.1038/msb.2008.78)

The job below is a **self-contained folder** — its own model, conf, data, reproduction figure, and
a `VALIDATION.md` scorecard grounding it in the primary sources.

## The biochemistry

A **radioligand competition / dissociation assay** on the IGF1 receptor. IGF1R is a pre-formed
disulfide-bonded **dimer** with two ligand sites (Site 1, Site 2) that a single IGF1 can
**crosslink** *within* the dimer (avidity), per the Kiselyov harmonic-oscillator mechanism.
Labelled ("hot") IGF1 is held fixed while unlabelled ("cold") IGF1 is titrated; the readout is
hot-ligand binding vs. cold dose. Erickson et al. fit **seven** rate constants (the eighth is fixed
by detailed balance) to **three** Kiselyov datasets at once.

## The job

| slug | fits | simulator | flavor | status |
|---|---|---|---|---|
| [`igf1r`](igf1r/) | 7 rate constants to **F5B + F5D_20min + F5D_60min** jointly (Kiselyov Fig 5B/5D) | **ODE** (finite network, no cap) | legacy (edition-1), in-model multi-phase `parameter_scan`, `chi_sq`, `normalization=init` → **NATIVE-ONLY** | ✅ validated 93/100 — see [`VALIDATION.md`](igf1r/VALIDATION.md) |

**Validated against the authors' own files.** Erickson's SI ("S2 Compressed File Archive") ships
the exact BioNetFit model + conf + data that produced the paper's **Table 1**; the job is built
from those, so its model is byte-identical to the published one, its data are the authors' own
Fig-5B/5D extractions, and it reproduces Erickson **Fig 3** at the Table-1 parameters.

## ⚠️ Re-scoped to match the literature (2026-07-11)

The `igf1r` slug **previously** carried a reduced **3-parameter** (`K1`/`K2`/`K1prime`), **F5B-only**
fit — a PyBNF teaching distillation (from the classic `examples/igf1r/`) that is **not** in
Erickson's paper. Validation against the primary source re-scoped it to Erickson's **actual**
published fit: **7 free rate constants + detailed balance, three datasets** (competition Fig 5B +
20/60-min dissociation Fig 5D). Same reaction network — the change is the parameterization, the
datasets, and the multi-phase protocol.

## ⚠️ Legacy (edition-1) mode — required

Unlike the edition-2 jobs in this corpus, `igf1r` runs in **legacy mode** (actions in the model,
datasets on the `model=` line). This is forced, not a shortcut: the F5D experiments need a stateful
2 h-preincubate → wash → cold-competition-scan protocol that PyBNF's edition-2 `experiment:`
directives cannot express. See the slug [`README`](igf1r/README.md) and `VALIDATION.md`.

## Source materials

- **Authors' fit files (primary):** Erickson 2019 SI **"S2 Compressed File Archive"** —
  `IGF1R_fit.bngl`, `IGF1R_fit.conf`, `F5B.exp`, `F5D_20min.exp`, `F5D_60min.exp` (the files that
  produced Table 1); used during validation but not redistributed here.
- **Model + data origin:** Kiselyov 2009 harmonic-oscillator model; Fig 5B (competition) and Fig 5D
  (20/60-min dissociation).
- **Related (do not confuse):** PyBNF's classic reduced `examples/igf1r/` job
  carry only the reduced 3-param / F5B-only distillation; RuleHub `Published/Mitra2019/15-igf1r/` is
  a *different* re-fit. This job matches the **paper**, not those corpus artifacts.

## Run

```bash
export BNGPATH="$HOME/Simulations/BioNetGen-2.9.3"   # folder with BNG2.pl
cd examples/real-world/Erickson-2019/igf1r
pybnf -c igf1r.conf
python make_reproduction.py
```
