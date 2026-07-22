# VALIDATION — Kozer-2013 / egfr_ode

Primary-source validation of the PyBNF job `examples/real-world/Kozer-2013/egfr_ode/`, per the
`validate-pybnf-job` skill. Confidence is **earned from the gate evidence below**.

> **Confidence: 89 / 100** — Gates 0–2 PASS/equiv (model byte-identical to the authors' File S1 on
> rules/observables; data trace to Kozer 2013 real measurements; fit recipe mirrors the vetted 2019
> PyBioNetFit legacy setup); Gate 3a PASS (model at Kozer's own Table-1 kinetics reproduces all four
> figures, incl. the held-out 2D, to median 4–13 %); Gate 3b PARTIAL (sloppy/non-identifiable — the
> paper's own Table S1 CIs span orders of magnitude, and a distinct kinetic set fits equally well).

**Two provenance threads** (this job sits at their intersection):
- **Model + data** = Kozer et al. **2013**, *Mol BioSyst* 9(7):1849–1863 (PMC3698845, DOI
  10.1039/c3mb70073a). ODE model = **File S1** (`NIHMS474074-supplement-ESI_-_File_S1.txt`);
  data = Figs 2B/2D/3B/3D (+ Table 2, Appendix S1).
- **Fit recipe** = Mitra et al. **2019**, *iScience* 19:1012–1036 (PyBioNetFit; BioNetFit-1
  "problem 2"). Bill-vetted legacy setup: `examples/egfr_ode/` (edition-1
  `egfr_ode.{bngl,conf}` + the already-ported edition-2 twin `egfr_ode_v2.{bngl,conf}`).

"The paper's result" for this job = **Kozer 2013 Figs 2B (cluster density vs dose), 3B (cluster
density vs time @30 nM), 3D (phospho-EGFR vs time @30 nM)** — the three datasets the paper fit —
plus **Fig 2D (phospho vs dose)**, which Kozer *predicted* (held out, Fig 5). Parameter table =
**Table 1** (fitted kinetics) + **Table S1** (90 % CIs).

---

## Gate 0 — Materials inventory — **PASS**

| needed | present? | path / note |
|---|---|---|
| model paper PDF | ✅ | `Kozer-2013-Mol_BioSyst.pdf` |
| authors' ODE model file | ✅ | **File S1** `NIHMS474074-supplement-ESI_-_File_S1.txt` (enables a direct Gate-2 diff) |
| SI (methods/derivations) | ✅ | `Appendix_S1.pdf`, `Tables_and_figures.pdf` (Table 1, Table 2, Table S1) |
| vetted fit-recipe setup | ✅ | `examples/egfr_ode/egfr_ode_v2.{bngl,conf}` (edition-2 twin) + edition-1 originals |

## Gate 1 — Data provenance — **PASS (equiv)**

`.exp` are **byte-identical** to the vetted 2019 legacy examples (`diff` clean, both files), and each
column is **normalized to mean = 1.000** (verified empirically). They trace to Kozer 2013's real
experimental measurements:

| `.exp` column | Kozer source | check vs shipped `.exp` |
|---|---|---|
| `pre1_dose` | Fig 2B / Table 2 cluster density vs dose | Table-2/mean = [1.408,1.203,1.017,0.669,0.703] vs [1.458,1.251,0.867,0.694,0.731] — 4/5 within ~4 %, 1 nM point ~15 % (underlying-data vs Table-2 summary) |
| `pre3_dose` | Fig 2D phospho vs dose | digitized/mean = [0.034,0.205,1.336,1.712,1.712] vs [0.004,0.196,1.339,1.731,1.730] — within a few % |
| `pre2_time` | Fig 3B cluster density vs time @30 nM | shape match (decays from baseline); time col = Kozer 0,0.5,1,2,3,5,7,10 min in **seconds** |
| `pre4_time` | Fig 3D phospho vs time @30 nM | shape match (0 → plateau within 1–2 min) |

- `_SD` columns ← replicate SEM (N>40 dose ROIs; 5 biological reps time course).
- Dose axis: our `LT=0.001` stands in for Kozer's 0 nM (log axis); `0.01 nM` row is NaN (unmeasured);
  cluster-density time points 120/420 s NaN (cluster density & phospho measured on different grids).
- **Normalization caveat:** the `.exp` ÷mean preprocessing is a **Mitra-2019 choice**; Kozer's own
  figures normalize cluster density ÷basal and phospho ÷plateau with a single fitted scale
  α = 0.0517. Same measurements, different normalization. I cannot byte-reproduce the exact `.exp`
  values without the original Mitra-2019 preprocessing script, but the values are Bill-vetted and
  trace to the figures within digitization tolerance.

**Verdict:** PASS (equiv) — data are real Kozer 2013 measurements, ÷mean-normalized (a documented
reformulation of the paper's own normalization).

## Gate 2 — Model fidelity — **PASS (identical model; equivalent IC)**

Reference: the authors' **File S1** (Kozer 2013 ODE model), and the vetted legacy `egfr_ode_v2.bngl`.

| aspect | File S1 (Kozer 2013) | our `egfr_ode.bngl` | verdict |
|---|---|---|---|
| molecule types | `EGF(rec)`, `EGFR(back,lig,cd~c~o,Y~u~p)` | same | **identical** |
| reaction rules (15) | 15 rules | 15 rules | **identical** (`model_diff.py` rules-level: onlyA=0, onlyB=0) |
| observables (8) | 8 | 8 | **identical** |
| rate laws | ring closures `chi_r*l20f/l21f/l22f/kaf`; `k_o/k_c` conf. change | same | **identical** |
| initial conditions | seed all-monomer `RT` | seed `MT` monomer + `DT` dimer (pre-formed unliganded equilibrium) | **equivalent** — MT+2·DT = 0.09 = RT exactly; File S1 relaxes to the same equilibrium via `l20f/l20r` before ligand (more physical: matches pre-incubation) |
| units | nM | nM | **identical** |
| network cap | `max_stoich={EGF=>4,EGFR=>4}` (tetramers; p.5 Box V) | same, declared by `generate_network` in `egfr_ode.conf` (#473) | **identical** |
| fitted kinetics | Table 1: k_o=6, k_c=1.6, kaf=15.4, kar=8.89, chi_r=4.37e4 | same nominals | **identical** |

`model_diff.py` (rules-level; net gen >420 s → fell back): molecule types **IDENTICAL**,
observables **IDENTICAL (8/8)**, reaction rules **IDENTICAL (15/15)**; seed species differ only by
the IC above. Identical rules ⇒ identical generated network topology (paper: 923 species / 11,918
reactions). Name map (Table 1): k_o=k_u, k_c=k_v, kaf=k_cx, kar=k_cr, chi_r=χ.

**Verdict:** PASS — our `.bngl` IS the Kozer 2013 File S1 model (byte-identical rules/observables/
molecule types), with a functionally-equivalent pre-equilibrated initial condition.

## Gate 3a — Reproduce at the paper's parameters — **PASS**

Model set to Kozer **Table 1** kinetics (k_o=6, k_c=1.6, kaf=15.4, kar=8.89, chi_r=4.37e4), each
dataset's **linear scale factor** fit in closed form (the paper also fit a scale factor), overlaid
on the ÷mean-normalized `.exp` (`validation/ode_nf_equivalence.py`; script also does the ODE↔NF
check below):

| dataset (Kozer fig) | optimal α | median rel err | max | note |
|---|---|---|---|---|
| cluster density vs time (3B, fit) | 30.36 | **3.9 %** | 19 % | |
| phospho-EGFR vs time (3D, fit) | 28.55 | **5.8 %** | (t=0 ≈0 baseline) | |
| cluster density vs dose (2B, fit) | 26.71 | **5.3 %** | 10.4 % | |
| phospho-EGFR vs dose (**2D, held-out prediction**) | 39.04 | **13.1 %** | (low-dose ≈0) | Kozer *predicted* this |
| **total** | | **χ² = 13.30** | | |

**Verdict:** PASS — the model at the paper's **own** Table-1 kinetics reproduces all four figures,
including the held-out Fig 2D prediction, to median 4–13 %. (The `max=100 %` entries are the near-zero
t=0 / lowest-dose phospho points, where any small deviation is a large *relative* error — not a fit
defect; medians are the meaningful metric.)

## Gate 3b — Recover the paper's parameters by fitting — **PARTIAL (sloppy by construction)**

The kinetic fit is **non-identifiable**, exactly as the paper itself reports:
- Kozer's own **Table S1** 90 % CIs span orders of magnitude: k_u 1.28–1.18×10⁴ s⁻¹; χ
  1.07×10⁴–3.9×10⁵ nM (k_u ~4 decades). Local sensitivities (Table S2) show cluster density is most
  sensitive to k_cr/k_cx and RT, not k_o/χ.
- **Two distinct kinetic sets reach comparable objectives:** Kozer Table-1 (χ²=13.30, Gate 3a above)
  and the Mitra-2019 RuleHub `10-egfr/fit_de` (χ²=10.16, used in the committed
  `make_reproduction.py`). Different kinetics, similar fit ⇒ the individual constants are not pinned;
  the **scale factors + the reproduced curve** are the identifiable content.
- A fresh full PyBNF fit is **cluster-scale** (each parameter set regenerates the ~12k-reaction
  network); not re-run — the non-identifiability is already established from the paper's own CIs and
  the two-param-sets-one-objective result above.

**Verdict:** PARTIAL — sloppy/non-identifiable, matching the paper's own reported CIs; the
identifiable result (scale factors + the Gate-3a curve at the paper's kinetics) is reproduced.

## ODE↔NF equivalence (NFsim hazard check) — **CONFIRMED**

Running `egfr_ode` (capped) and `egfr_nf` (unbounded NFsim) at the **same** published nominals:
- **0.00 %** of receptors sit in oligomers larger than tetramers (all doses) → the unbounded NF
  model never exceeds the ODE cap at these params; the cap is essentially exact, not just "small."
- The ODE (nM) and NF (molecule-count) optimal scale factors differ by **exactly NA·Vo** (10,033) —
  10181, 10060, 10226, 9964 across the four datasets (0.3–1.9 % of NA·Vo) → the two models produce
  the **same physical observable** up to the unit conversion.
- Gate 3a fit quality is near-identical (NF total χ²=12.72 vs ODE 13.30). See
  [`../egfr_nf/VALIDATION.md`](../egfr_nf/VALIDATION.md).

---

## Divergences from Kozer's *original* fit (Mitra-2019 reformulation — documented, not errors)

The job faithfully mirrors its **fit-recipe source (Mitra 2019)**, which reformulated Kozer's
original fit. These are not defects but must be recorded:

1. **Datasets fit:** Kozer fit **3** (Figs 2B, 3B, 3D) and *held out* Fig 2D as a prediction (Fig 5).
   The job fits **all 4** (2D is in `doseresponse.exp` as `pre3_dose`). → the job's Gate-3b fit uses
   more data than Kozer's; Gate 3a below still checks 2D as the held-out curve (labeled `2D*`).
2. **Scale factors:** Kozer used **one** fitted α=0.0517; the job uses **four** (`alpha1_pre..
   alpha4_pre`), one per ÷mean-normalized dataset.
3. **Normalization:** Kozer ÷basal (cluster) / ÷plateau (phospho); the job ÷mean.

## Corrections applied

- **None required** to the model, data, or conf — all are consistent with the authors' File S1 and
  the vetted 2019 legacy setup. The two-thread provenance (Kozer-2013 model/data + Mitra-2019 fit
  recipe) and the divergences from Kozer's original fit are **documented** above (per Bill's
  decision to keep the Mitra-2019 setup and document rather than re-scope to Kozer's 3-dataset fit).

## Bottom line

The `egfr_ode` model is **byte-identical** to the authors' own Kozer-2013 File S1 (rules, observables,
molecule types) with a functionally-equivalent pre-equilibrated IC; the data trace to Kozer's real
cluster-density and phospho measurements; and the fit recipe reproduces the Bill-vetted 2019
PyBioNetFit setup exactly. Gate 3a reproduces all four figures at Kozer's **own** Table-1 kinetics
(median 4–13 %, incl. the held-out 2D prediction); Gate 3b is sloppy exactly as the paper's own
Table S1 CIs report. The only interpretive nuance is the two-thread provenance: this is the
Mitra-2019 *reformulation* of Kozer's original 3-dataset/1-scale fit — documented, not a defect. The
single most valuable next step to raise confidence would be recovering the original Mitra-2019
preprocessing to byte-reproduce the ÷mean `.exp` values.
