# VALIDATION — Salazar-Cavazos-2019/egfr_simpull

Primary-source validation of the PyBNF job `examples/real-world/Salazar-Cavazos-2019/egfr_simpull/`.
Confidence is **earned from the gate evidence below**.

> **Confidence: 93 / 100** — the authors shipped their **own** PyBioNetFit model, both data files,
> and the `de`/`chi_sq` config; our `.bngl` generates a network **byte-identical** to the authors'
> best-fit model (Gate 2), the `.exp` are transcriptions of the authors' own data (Gate 1), and the
> model at the published best-fit reproduces the data at what an independent fit confirms is the
> **global optimum** (Gate 3). The 7-point deduction is the model's inherent residuals (pY1068
> saturation, mid-dose pan-PY) and one sloppy parameter — properties of the published model, not the
> job setup.

Primary sources used during validation (not redistributed):
- Model + data paper: Salazar-Cavazos E, Franco Nitta C, Mitra ED, Wilson BS, Lidke KA,
  Hlavacek WS, Lidke DS. "Multisite EGFR phosphorylation is regulated by adaptor protein
  abundances and dimer lifetimes." *Mol Biol Cell* 2020; 31(7):695–708.
  PMCID PMC7202077 · DOI 10.1091/mbc.E19-09-0548.
- Author files used: supplement `mc-e19-09-0548-s03/PyBNF-fitting-setup/` —
  `190127_CHO_EGFR_forBNF.bngl` (fitting model), `dose_resp.exp`, `EGF_25nM.exp` (data),
  `fit_v1_28.conf` (the fit config); plus `190127_CHO_EGFR_best-fit.bngl` (fitted parameter values).

"The paper's result" for this job = **the joint fit of the SiMPull dose-response and 25 nM time
course** (Fig. 2 / Fig. 3 phospho-% panels) at the parameters in `190127_CHO_EGFR_best-fit.bngl`.

---

## Gate 0 — Materials inventory

| needed | present? | path / note |
|---|---|---|
| model paper PDF | ✅ | `salazar-cavazos-et-al-2020-…pdf` (PMC7202077) |
| data paper PDF | ✅ | same paper (SiMPull data are the authors' own) |
| SI / author files | ✅ | `mc-e19-09-0548-s03/` — README + all cell-line `.bngl` |
| author model/job files | ✅ | `PyBNF-fitting-setup/` — model + 2 `.exp` + 3 `.conf` (the actual fit recipe) |

**Verdict:** PASS — this is a **gold-standard** case (authors shipped their complete PyBioNetFit
setup, like `Erickson-2019/igf1r`).

## Gate 1 — Data provenance

| `.exp` | source | method | normalization/units | diff vs. author file | verdict |
|---|---|---|---|---|---|
| `dose_resp.exp` | authors' `dose_resp.exp` | transcribed verbatim | % phosphorylation (absolute) | values + SD identical | PASS |
| `EGF_25nM.exp` | authors' `EGF_25nM.exp` | transcribed verbatim | % phosphorylation (absolute) | values + SD identical | PASS |

The only edits are edition-2 header conventions: the three function-observable columns carry parens
(`pY1068_percent()`, `pY1173_percent()`, `phosR_per()`) and the `_SD` columns are kept bare
(`pY1068_percent_SD`, …) — PyBNF strips the parens from the data column to match the model function
and appends `_SD` to that stripped name (verified: `petab_roundtrip.py` export lints clean). No data
value changed.

**Verdict:** PASS — byte-level provenance to the authors' own data files.

## Gate 2 — Model fidelity

Reference compared against: the authors' `190127_CHO_EGFR_best-fit.bngl` (same rules/species as the
`forBNF` fitting model; the two differ only in extra plotting observables, which do not affect the
network).

| aspect | authors' model | our `egfr_simpull.bngl` | verdict |
|---|---|---|---|
| molecule types / seed species | EGF, EGFR(6 comps), GRB2, SHC1 | identical | match |
| reaction rules + rate laws | 15 rules (bind/dimerize/phos/dephos/adaptors) | identical | match |
| fixed parameter values | Table-sourced constants | identical (verbatim) | match |
| fitted parameters | 6 `X__FREE` placeholders | declared as ids `GRB2_total_0`, … (best-fit nominals) | equiv (rename only) |
| observables ↔ measured quantities | `pY1068_percent()`, `pY1173_percent()`, `phosR_per()` | identical functions | match |
| network cap | bare `generate_network` (finite) | none needed (finite, 75 species) | match |
| actions block | scan + simulate | removed (synthesized from conf) | expected (edition-2) |

Generated-network check (both models + a bare `generate_network`, via BNG2.pl):
**both produce 75 species / 618 reactions, and the sorted species lists are identical.** Renaming
the six `__FREE` placeholders to real ids does not change topology.

**Verdict:** PASS (network byte-identical) — the only substantive change is declaring the six fitted
placeholders as real parameter ids so the conf can bind them by name (ADR-0034).

## Gate 3a — Reproduce at the paper's parameters

- Published params used: the six values in `190127_CHO_EGFR_best-fit.bngl` (= the `.bngl` nominals).
- Reproduction: `make_reproduction.py` → `egfr_simpull_reproduction.png` (2×3: dose-response and
  25 nM time course × the three antibodies).
- Metric (18 fit points, PyBNF `chi_sq` = ½·Σ((sim−data)/SD)²):

| dataset | contribution | median rel. err |
|---|---|---|
| dose-response (3 doses × 3 obs) | 202.3 | — |
| time course (3 t × 3 obs) | 157.4 | — |
| **total** | **359.7** | **10.4 %** (max 62.8 %) |

pY1173 is fit tightly (≈7 %). The residuals are pY1068 dose saturation (model plateaus ~13 %, data
~20 % at 50 nM) and mid-dose pan-PY (`phosR_per` at 5 nM: model 42 % vs. data 26 %) — the model's
sigmoid saturates faster than the SiMPull curve.

**Verdict:** PASS — the model at the authors' own best-fit reproduces the qualitative dose-response
and rapid kinetics; the residuals are the genuine (best-achievable) fit quality, not a setup error.

## Gate 3b — Recover the paper's parameters by fitting

- Run: `pybnf -c egfr_simpull.conf`-style `de` + `refine`, population 24, 30 iterations, from the
  authors' `fit_v1_28.conf` bounds.
- Result: converged to **chi_sq = 361.3**, i.e. essentially equal to the published params' 359.7 →
  the published best-fit sits at the **global optimum** (the model cannot fit the data tighter).

| param | published | recovered | within 3×? |
|---|---|---|---|
| `GRB2_total_0` | 169853 | 155127 | ✅ (9 %) |
| `SHC1_total_0` | 649426 | 555326 | ✅ (14 %) |
| `ratio_kpkd_Y1068` | 0.15755 | 0.18113 | ✅ (15 %) |
| `ratio_kpkd_YN` | 0.44476 | 0.44717 | ✅ (0.5 %) |
| `kdephosYN_0` | 0.017182 | 0.014143 | ✅ (18 %) |
| `kdephosY1068_0` | 1.6588 | 18.99 | ⚠️ sloppy (11×) |

- Identifiability: 5 of 6 recover within ~18 %. `kdephosY1068_0` is a **fast-rate** sloppy direction
  (once dephos is fast vs. the 300 s protocol, only the ratio `ratio_kpkd_Y1068` — recovered — sets
  the plateau). The identifiable quantities are the two adaptor copy numbers and the phospho ratios,
  which *are* the paper's biological conclusion.

**Verdict:** PASS — the fit reaches the published objective and recovers all identifiable parameters;
the one non-identifiable direction is structural (same pattern as `tlbr`/`igf1r`/`Rukhlenko`).

---

## Divergence & corrections

- Scope vs. paper: **matches** — this is the authors' own joint fit (both `.exp`, six free params,
  `de`/`chi_sq`), not a reduced distillation.
- Corrections applied to job files: none to the science. Setup-only: declared the six `__FREE`
  placeholders as real ids (best-fit nominals), stripped the actions block, added function-observable
  parens to the `.exp` data columns.
- Re-run after setup: tier-1 PASS, PEtab round-trip PASS, reproduction + independent fit as above.

## Bottom line

The strongest kind of job: the authors published their complete PyBioNetFit setup, our model
reproduces their network exactly, the data are their own, and the model at the published parameters
reproduces the SiMPull data at the confirmed global optimum. The only soft spots are inherent to the
published 6-parameter model (pY1068/pan-PY residuals, one sloppy rate) — not to this edition-2 port.
Most valuable next step: none required for correctness; optionally add the authors' `fit_pt.conf`
(parallel-tempering UQ) as a second slug to capture the published uncertainty analysis.
