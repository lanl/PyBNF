# Kirsch-2020 — ATF2 phosphoswitch, JNK & p38 co-regulation (PyBNF fitting jobs)

A selected PyBNF edition-2 parameter-fitting job derived from one paper:

> Kirsch K, Zeke A, Tőke O, Sok P, Sethi A, Sebő A, Kumar GS, Egri P, Póti ÁL,
> Gooley P, Peti W, Bento I, Reményi A, Alexa A.
> **"Co-regulation of the transcription controlling ATF2 phosphoswitch by JNK and p38."**
> *Nat Commun* 2020; **11**(1):5769.
> PMCID: [PMC7666158](https://pmc.ncbi.nlm.nih.gov/articles/PMC7666158/) ·
> DOI: [10.1038/s41467-020-19582-3](https://doi.org/10.1038/s41467-020-19582-3)

The selected job is a **self-contained folder** with its own models, configuration,
constraints, reproduction figure, and detailed README.

## The shared model

The job reconstructs the **rule-based model** that the authors provide as
**Supplementary Software 1** (`Bionetgen_JNK-p38-ATF2_model.bngl`; its actions block
generates the paper's Fig. 7b). JNK and p38 dock the ATF2 transactivation domain at
distinct sites — JNK at the D-site, p38 bipartitely at the D-site **and** the F-site
(SPFENEF / S90 region) — and distributively phosphorylate the T69/T71 "phosphoswitch". JNK
additionally phosphorylates the F-site **S90**, and **S90-P sterically blocks p38's F-site
recruitment** — the paper's central mechanism. Deterministic ODE, **48 species, 152
reactions**. The edition-2 reconstruction reproduces the authors' trajectories **exactly**
(max relative difference 0.0).

## The selected job

| slug | fits | flavor | data source | status |
|---|---|---|---|---|
| [`phosphoswitch_bpsl`](phosphoswitch_bpsl/) | S90 phosphoswitch → p38 recruitment orderings (4 cell params) | **BPSL** constraints, **native-only** | Suppl. Table 2 mutants; Figs. 3c/4b binding | ✅ tier-1 + **`check` 6/6 satisfied** (build-verified; not yet primary-source-audited) |

`phosphoswitch_bpsl` expresses the paper's central *qualitative* claim—JNK's S90
phosphorylation diminishes p38 binding:
**S90N > JNK inhibitor > WT > MUT4** for p38:ATF2 recruitment as PyBNF BPSL `.prop`
constraints.
The constraint-only fit is native-only because BPSL has no PEtab representation. Its
published-parameter check satisfies all six constraints, and the bounded fit exercises
the full simulate → qualitative-score → propose loop across four condition-specific models.

## Source materials

All from the article's supplementary files (Springer CDN); no public GitHub / BioModels
deposition exists.

| file | content |
|---|---|
| [Supplementary Software 1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-19582-3/MediaObjects/41467_2020_19582_MOESM4_ESM.zip) | the authors' BioNetGen `.bngl` (WT parameters) |
| [Supplementary Information PDF](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-19582-3/MediaObjects/41467_2020_19582_MOESM1_ESM.pdf) | **Supplementary Table 2** (all fitted parameters + data provenance); Fig. S7 in-vitro kinetics |

## Run

```bash
export BNGPATH="$HOME/Simulations/BioNetGen-2.9.3"   # folder with BNG2.pl
cd examples/real-world/Kirsch-2020/phosphoswitch_bpsl
pybnf -c phosphoswitch_bpsl.conf
```

See each slug's `README.md` for its full write-up.
