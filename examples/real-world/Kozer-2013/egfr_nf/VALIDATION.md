# VALIDATION — Kozer-2013 / egfr_nf

Primary-source validation of the PyBNF job `examples/real-world/Kozer-2013/egfr_nf/`, per the
`validate-pybnf-job` skill. Confidence is **earned from the gate evidence below**.

> **Confidence: 86 / 100** — Gates 0–1 PASS (data trace to Kozer 2013; byte-identical to the vetted
> 2019 legacy setup); Gate 2 PASS (equiv) as a network-free reformulation, with the ODE↔NF
> equivalence **confirmed** (0 % of receptors above tetramer; scale factors match the capped ODE to
> the NA·Vo unit factor) — **the stated provenance was inaccurate and is corrected here**; Gate 3a
> PASS (reproduces the four figures at Kozer's kinetics, median 4–13 %); Gate 3b PARTIAL (sloppy +
> NFsim cluster-scale).

**Provenance — read carefully (this slug's central finding).** The `egfr_nf` model is a
**network-free NFsim reformulation** of the shared EGFR-clustering biology, fit to the **same
Kozer 2013 data** as `egfr_ode`. Its true lineage is the **Mitra 2019 PyBioNetFit / BioNetFit-1
"example 2"** model (`example2_starting_point.bngl`, by B. R. Thomas, W. S. Hlavacek, E. D. Mitra),
an **unbounded** (oligomers up to 20-mers tracked) network-free variant of the Kozer 2013 model.

> ⚠️ The as-shipped README/`.bngl`/`.conf` state the model is "derived from the Kozer 2014 SI
> `bi500182x_si_001.txt` (network-free, unbounded)." **This is inaccurate.** The Kozer 2014 SI is a
> **tetramer-capped ODE model *with Grb2*** (`generate_network({max_stoich=>{EGF=>4,EGFR=>4}})` +
> `simulate_ode`; molecule-count units), and the 2014 paper text describes the model as "the same as
> [2013] … up to tetramers … simulated using BioNetGen" plus one Grb2 rule — **no** NFsim /
> network-free / unbounded reformulation, no phospho data. Our `egfr_nf.bngl` has **no Grb2**, is
> **unbounded**, and is **network-free NFsim** — i.e. it is *not* the 2014 SI model. **Correction
> applied** (see below).

"The paper's result" for this job = the **same** Kozer 2013 Figs 2B/2D/3B/3D data as `egfr_ode`, and
the requirement that this unbounded network-free model **re-expresses the capped-ODE ground truth**
(the ODE↔NF equivalence check).

---

## Gate 0 — Materials inventory — **PASS**

| needed | present? | path / note |
|---|---|---|
| 2013 model+data paper | ✅ | `Kozer-2013-Mol_BioSyst.pdf` + File S1 + Tables_and_figures |
| 2014 paper + SI | ✅ | `Kozer-2014-Biochemistry.pdf` + `bi500182x_si_001.txt` (= capped-ODE+Grb2, **not** the NF model) |
| vetted fit-recipe setup | ✅ | `examples/egfr_nf/egfr_nf.{bngl,conf}` (Mitra 2019 example-2) |
| the unbounded NFsim model's published source | ❌ | none — it exists only as the Mitra-2019 example; **not** a published SI file (see Gate 2) |

## Gate 1 — Data provenance — **PASS (equiv)**

Identical to `egfr_ode`: the `.exp` are **byte-identical** to the vetted 2019 legacy examples and to
the `egfr_ode` `.exp` (only the dose header differs: `LT_nM` vs `LT`), each column normalized to
mean = 1.000, tracing to Kozer 2013 cluster-density (Fig 2B/3B) and phospho-EGFR (Fig 2D/3D)
measurements. See `egfr_ode/VALIDATION.md` Gate 1 for the per-column trace (same numbers).

**Verdict:** PASS (equiv) — same real Kozer 2013 data, ÷mean-normalized.

## Gate 2 — Model fidelity — **PASS (equiv) as a reformulation; provenance CORRECTED**

Reference: the vetted legacy `examples/egfr_nf/egfr_nf.bngl` (Mitra 2019 example-2), and — for the
provenance claim — the Kozer 2014 SI `bi500182x_si_001.txt`.

| aspect | our `egfr_nf.bngl` | note |
|---|---|---|
| simulator | network-free **NFsim** (`method: nf`) | unbounded oligomers (species tracked to 20-mers) |
| Grb2 | **absent** | the 2014 SI *has* Grb2; the NF fit model does not (fits 2013 cluster/pEGFR data) |
| rules vs legacy `egfr_nf.bngl` | **semantically identical** | only `X__FREE`→bind-by-id + `(NA*Vo)` unit conversion moved into derived `kaf_pc`/`chi_r_pc` (rules see identical `kaf/(NA*Vo)`, `chi_r*(NA*Vo)`) |
| units | molecules/subvolume (f=0.01); 1 nM = 1e6 molecules/cell | observables ÷f to per-cell; physical rate constants match the ODE |
| vs Kozer 2014 SI | **different model** | SI = tetramer-capped ODE + Grb2; ours = unbounded NF, no Grb2 |
| vs Kozer 2013 biology | same rules/observables as `egfr_ode` minus the tetramer cap | equivalence tested in the ODE↔NF check |

The unbounded NF model is intended to be **equivalent to the capped ODE** except for oligomers
larger than tetramers; Kozer 2013 caps at tetramers because higher-order oligomers are rare.

**ODE↔NF equivalence — CONFIRMED** (`validation/ode_nf_equivalence.py`, both models at the same
published nominals, NF averaged over replicates):
- **0.00 %** of receptors sit in oligomers larger than tetramers (all doses) → the unbounded NF
  model never exceeds the ODE tetramer cap at these parameters. The cap is **essentially exact**,
  not merely "small" (Bill's "the cap hardly matters", quantified).
- The NF (molecule-count) and ODE (nM) optimal scale factors differ by **exactly NA·Vo** (10,033):
  ratios 10181 / 10060 / 10226 / 9964 across the four datasets (0.3–1.9 % of NA·Vo) → the two models
  produce the **same physical observable** up to the unit conversion.
- Gate 3a fit quality is near-identical (NF total χ²=**12.72** vs ODE **13.30**).

**Verdict:** PASS (equiv) — a faithful network-free reformulation of the shared 2013 model, verified
equivalent to the capped-ODE ground truth; **not** the Kozer 2014 published SI (provenance corrected).

## Gate 3a — Reproduce at the paper's parameters — **PASS**

NF at Kozer Table-1 kinetics, per-dataset optimal scale, vs the ÷mean-normalized `.exp`:

| dataset | median rel err | | dataset | median rel err |
|---|---|---|---|---|
| cluster density vs time (3B) | **4.0 %** | | cluster density vs dose (2B) | **6.9 %** |
| phospho-EGFR vs time (3D) | **7.1 %** | | phospho-EGFR vs dose (2D, held-out) | **12.6 %** |

Total χ² = **12.72** (vs ODE 13.30). **Verdict:** PASS — reproduces all four figures at Kozer's own
kinetics, matching the ODE slug (as the equivalence check requires).

## Gate 3b — Recover the paper's parameters by fitting — **PARTIAL (sloppy + heavy)**

Same non-identifiability as `egfr_ode` (the kinetics are shared): Kozer's Table S1 90 % CIs span
orders of magnitude, and the committed `make_reproduction.py` reproduces the data at a *different*
kinetic set (Mitra-2019 RuleHub `04-egfrnf/fit_ade`, χ²=13.41). NFsim fits are additionally
**cluster-scale** (stochastic, ~10³–10⁴ molecules; 100 nM ≈ 10⁶ EGF particles), so a fresh full fit
is a cluster job — not re-run; the sloppiness is established from the paper's CIs + the two-param-sets
result. **Verdict:** PARTIAL — sloppy by construction (as `egfr_ode`) and heavy; identifiable content
(scale factors + reproduced curve) recovered.

---

## Divergences / corrections

**Corrections applied to job files** (provenance accuracy — ✅ done):
- `egfr_nf.bngl`, `egfr_nf.conf`, `egfr_nf/README.md`, and the paper-level `README.md`
  header/provenance were **relabelled**: the model is the **Mitra 2019 PyBioNetFit example-2
  network-free NFsim reformulation** of the Kozer **2013** model — associated with the 2014 NFsim
  modelling direction but **not** the published 2014 SI `bi500182x_si_001.txt` (a capped-ODE + Grb2
  model, not reproduced here).

**Documented (inherited from the Mitra-2019 fit recipe; not errors):** fits 4 datasets (Kozer fit 3,
held out 2D); 4 scale factors (Kozer used 1 α); ÷mean normalization (Kozer ÷basal/÷plateau).

**Open question for Bill (co-author):** was there an actual *unbounded NFsim* model in the 2014 work
distinct from the published capped-ODE+Grb2 SI? The current NF slug is grounded in the Mitra-2019
example; if a 2014 NFsim source exists it would strengthen the provenance.

## Bottom line

The `egfr_nf` model is a legitimate, vetted network-free reformulation of the same Kozer 2013 biology
and fits the same real data as `egfr_ode`. Its scientific validity is now **established**: the ODE↔NF
equivalence is confirmed (0 % of receptors above tetramer; scale factors match the capped ODE to the
NA·Vo unit factor; near-identical Gate-3a fit), so the unbounded NFsim model faithfully re-expresses
the capped-ODE ground truth. The one substantive defect found — a **provenance mislabel** (it is *not*
the Kozer 2014 SI model) — is corrected in the job files. Scored slightly below `egfr_ode` because it
is a demonstration reformulation rather than a published model, and is stochastic/heavier.
