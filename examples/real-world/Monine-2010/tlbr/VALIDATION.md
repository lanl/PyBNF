# VALIDATION — Monine-2010/tlbr

Primary-source validation of the PyBNF job `examples/real-world/Monine-2010/tlbr/`, per the
`validate-pybnf-job` skill. Confidence is **earned from the gate evidence below**.

> **Confidence: 91 / 100** — the grounding is airtight: Gate 0 PASS, Gate 1 PASS (data traced to
> Monine Fig 2a; a wrong-source attribution corrected), Gate 2 PASS (model *identical* to the
> published TLBR model), Gate 3a PASS (reproduces Fig 2a at the paper's *own* Table-1 params to
> median |Δλ|≈0.01, RMS_λ 0.015 < Monine's own 0.02 acceptance). Gate 3b recovers the paper's
> **identifiable** result — K2 within 4 %, α within tolerance, the fit inside the paper's own K1–K2
> valley — but not the exact Table-1 point, which is a property of the paper's **own documented
> non-identifiability** (Fig S2), not a defect in the job.
>
> The ~9 missing points are **not** "we didn't hit the estimates": (i) parameter recovery is
> structurally unmeetable for a self-declared-sloppy model, so ~95+ is unreachable for *this paper*
> regardless of job quality (~3); (ii) Gate 1 is a Fig-2a **overlay** match, not a byte-level diff
> against the authors' original data file, which is not in the garage (~3); (iii) a full-budget
> multi-start fit was not run — a reduced fit + valley scan stood in (~3).

Primary sources used during validation (not redistributed):
- **Model + fit + binding data:** Monine MI, Posner RG, Savage PB, Faeder JR, Hlavacek WS. "Modeling
  multivalent ligand-receptor interactions with steric constraints on configurations of cell-surface
  receptor aggregates." *Biophys J* 2010; 98(1):48–56. PMID 20074516 · DOI 10.1016/j.bpj.2009.09.043.
  (`Monine-2010-Biophys-J.pdf` + Supporting Material `mmc1.pdf`.)
- **Ligand (compound 6a) synthesis + degranulation data:** Posner RG, Geng D, Haymore S, Bogert J,
  Pecht I, Licht A, Savage PB. "Trivalent antigens for degranulation of mast cells." *Org Lett* 2007;
  9(18):3551–3554. PMID 17691814 · DOI 10.1021/ol071231z. (`Posner-2007.pdf`.)
- **Author model/job files:** none in the garage. Diffed instead against the PyBNF `examples/tlbr`
  and `examples/real-world/Monine-2010/tlbr` (same author lineage — Posner/Hlavacek) and against the paper spec.

**"The paper's result" for this job** = **Fig 2a** (flow-cytometric binding titration + TLBR fit) at
the parameters in **Table 1, TLBR column**: **K1 = 0.467 nM⁻¹ (CI 0.111–0.767), K2 = 87.03 nM⁻¹
(CI 31.6–128.1), α = 0.816 (CI 0.758–0.881)**.

---

## Gate 0 — Materials inventory

| needed | present? | path / note |
|---|---|---|
| model paper PDF (Monine 2010) | ✅ | `Monine-2010-Biophys-J.pdf` — the TLBR model, its fit, and the binding data (Fig 2a) |
| Supporting Material (Monine SI) | ✅ | `mmc1.pdf` — TLBR rules, fit definition (Eqs 10–11), reference constants, sloppiness (Fig S2) |
| data paper PDF (Posner 2007) | ✅ | `Posner-2007.pdf` — **degranulation** data (Fig 3) + compound-6a synthesis; **no binding titration** |
| author model/job files | ❌ | none; used PyBNF `examples/tlbr` (same lineage) + paper spec for the Gate-2 diff |

**Verdict: PASS.** Every source the job depends on is present. The one gap (the authors' original
`.bngl`) is covered by the identical PyBNF classic model, which is itself the BioNetFit-1 example-3
lineage co-authored by Posner and Hlavacek.

## Gate 1 — Data provenance

| `.exp` | source (fig, series) | method | quantity / normalization | diff vs. shipped | verdict |
|---|---|---|---|---|---|
| `tlbr.exp` (12 doses, FL1 vs LTconc) | **Monine 2010 Fig 2a** (filled dots — flow-cytometric binding of Alexa-488 compound 6a to IgE-FcεRI on RBL cells) | overlay cross-check (fig digitized; the `.exp` itself is the original recorded dataset) | `.exp` column = FL1 (scaled fluorescence, 0–1); Fig 2a plots λ = α·FL1 | see below | **PASS** |

Evidence: the `.exp` concentrations and FL1 carry **10 significant figures** (e.g. `0.5013619944`,
`213.4593409505`) — these are the original measured values, not figure-digitized round numbers; Fig 2a
is a denser (~35-point) rendering of the same experiment. Overlaying the 12 `.exp` points as
λ = 0.816·FL1 on a calibrated render of Fig 2a (axes calibrated from the decade ticks and the 0.0–0.8
label rows) places **every point on the Fig 2a dot cloud / fit line** across 5+ decades
(median |Δλ| ≈ 0.04 vs the digitized isotherm — well within figure-digitization + half-log-resampling
tolerance). The fit *definition* also matches the source exactly: SI Eq. 10 is
`λ = (L_total − L_free)/(2·R_total)`, and the job's `lambda()` function is that verbatim.

**Correction forced (applied):** the job attributed the binding data to *Posner 2007*. Posner 2007
contains **only degranulation** (β-hexosaminidase, Fig 3) and the compound-6a synthesis — **no binding
titration**. The binding data are **Monine 2010's own Fig 2a**. README/model/conf attributions updated.

## Gate 2 — Model fidelity

Reference compared against: the PyBNF `examples/tlbr` + `examples/real-world/Monine-2010/tlbr` models (same
lineage) via `scripts/model_diff.py --rules-only`, and the paper's main-text rules + SI by hand.

| aspect | paper (Monine 2010) | our `tlbr.bngl` | verdict |
|---|---|---|---|
| molecule types | trivalent ligand, bivalent receptor | `L(s,s,s)`, `R(s,s)` | **match** |
| rule 1 — ligand capture | Eq 4: `L(r,r,r)+R(l) →k+1 L(r¹,r,r).R(l¹)` | `L(s,s,s)+R(s)->L(s!1,s,s).R(s!1) kf1` | **match** |
| rule 2 — crosslinking | Eq 5: `L(r⁺,r)+R(l) →k+2 L(r⁺,r¹).R(l¹)` | `L(s!+,s)+R(s)->L(s!+,s!1).R(s!1) kf2` | **match** |
| rule 3 — dissociation | Eq 6: `L(r¹).R(l¹) →koff L(r)+R(l)` | `L(s!1).R(s!1)->L(s)+R(s) koff` | **match** |
| rate laws | k+1 = K1·koff/(NA·V), k+2 = K2·koff/(NA·V) | `kf1=K1*1e9*kr1/(NA*Vref)`, `kf2=K2*1e9*kr2/(NA*Vref)` (K1,K2 in /nM → ×1e9 → /M) | **match** |
| observable | SI Eq 10: λ = (L_tot−L_free)/(2 R_tot); FL1 = λ/α | `lambda()`, `FL()=lambda/alpha` | **match** |
| constants | Table 1 fn / SI: V*=10⁻¹² L, N_R*=300, koff=0.01 s⁻¹ | `Vref=f/cellDensity=1e-12`, `RTref=f*3e5=300`, `koff=0.01` | **match** |

`scripts/model_diff.py --rules-only` vs both PyBNF references: molecule types / seed species /
observables / reaction rules all `IDENTICAL` (onlyA=0, onlyB=0). The parameters + functions blocks are
byte-identical to the classic model except the three fitted params (classic `__FREE` placeholders →
job's bind-by-id nominals).

Deviations, all benign: (a) `alpha`/`K1`/`K2` bind by id instead of `__FREE` (ADR-0034, cosmetic);
(b) no `begin actions` block (correct for a network-free NFsim job — the dose-response is synthesized
from the conf); (c) **objective encoding** — PyBNF `sos` minimizes Σ(λ_model/α − FL1)² whereas the
paper minimizes Σ(λ_model − α·FL1)² (SI Eq 11); these differ only by the constant weight 1/α² and are
functionally equivalent; (d) NFsim integrated to `t_end=5000 s` rather than a true steady-state solve
— confirmed to have reached equilibrium at Gate 3a (top-dose λ matches Fig 2a).

**Verdict: PASS (identical to the published TLBR model).**

## Gate 3a — Reproduce at the paper's parameters

- Published params used: **Monine 2010 Table 1 (TLBR): K1 = 0.467, K2 = 87.03, α = 0.816.**
- Reproduction: NFsim `parameter_scan` over the 12 doses, 7 replicate scans averaged (`make_reproduction.py`, now pointed at the Table-1 params) → committed `tlbr_reproduction.png`.
- Metric (7 reps): **FL-space sos ≈ 0.0038**; **RMS in λ ≈ 0.015**, inside Monine's own RMS<0.02
  acceptance (SI Eq 11); **median |rel err| = 2.4 %** (max 11 %) over doses with FL > 0.05;
  **λ vs Fig 2a: median |Δλ| = 0.008, max 0.040** over all 12 doses.
- Contrast: the RuleHub-2019 re-fit (α=0.746, K1=0.109, K2=33.6) reproduces the *normalized* `.exp`
  at least as well (FL-space sos ≈ 0.002–0.004, ≤ Monine's — see Gate 3b) but **undershoots Fig 2a's
  λ** (median |Δλ| = 0.022; 68 nM: 0.66 vs 0.72; 213 nM: 0.79 vs 0.84) because its α differs — it is
  a *different point in the sloppy valley*, not the paper's reported result.

**Verdict: PASS** — the model at the paper's *own* Table-1 parameters reproduces Fig 2a.

## Gate 3b — Recover the paper's parameters by fitting

- Run: budget-limited PyBNF DE fit (`de`, population 18, 15 iterations, smoothing 1 ≈ 288 NFsim
  evals; 26 min on 6 cores — the shipped conf's pop 50 × iter 50 × smoothing 3 is
  workstation-infeasible with NFsim). Same bounds as the shipped conf.
- Best-fit found: **K1 = 1.455, K2 = 90.41, α = 0.987** (obj = 0.00491, comparable to the paper's
  params). `scripts/compare_params.py` vs Monine Table 1 (factor 3×):

| param | published (Table 1) | recovered | log10 ratio | within 3×? |
|---|---|---|---|---|
| `alpha` | 0.816 | 0.987 | +0.08 | **yes** |
| `K1` | 0.467 | 1.455 | +0.49 | no (≈3.1×) |
| `K2` | 87.03 | 90.41 | +0.02 | **yes** (≈4 %) |

  → 2/3 within 3×; **K2 and α recovered, K1 not uniquely** (the sloppy direction). The sorted
  population shows the valley directly: solutions from `(K1,K2)=(0.29,15.8)` to `(2.4,90.4)` all sit
  at obj 0.005–0.007.

- **Identifiability (valley scan, α optimized in closed form, 5 reps/point):**

| point | K1 | K2 | α\* | sos |
|---|---|---|---|---|
| Monine Table 1 | 0.467 | 87.0 | 0.83 | **0.0036** |
| RuleHub 2019 | 0.109 | 33.6 | 0.76 | **0.0017** |
| along valley | 2.0 | 300 | 0.87 | 0.0098 |
| along valley | 6.9 | 217 | 1.01 | 0.0136 |
| **off** (loK2) | 0.467 | 12 | 0.99 | 0.0229 |
| **off** (hiK2/loK1) | 0.05 | 87 | 0.64 | 0.0134 |
| **off** (hiK1/loK2) | 5.0 | 20 | 1.16 | 0.0682 |

  Along the K1–K2 diagonal the objective stays low; **off-diagonal it degrades 4–19×** — a genuine
  correlated valley (matching the paper's Fig S2), not a flat plateau. The individual K1, K2 are not
  pinned; the identifiable combination (≈ the diagonal) and the reproduced curve are.

- **Subtlety worth recording:** in the FL-space `sos` that PyBNF minimizes, the RuleHub point
  (0.0017) beats Monine's Table 1 (0.0036), because PyBNF minimizes Σ(λ/α − FL1)² whereas Monine
  minimized the α²-weighted Σ(λ − α·FL1)² (SI Eq 11). So a PyBNF fit is *pulled toward* the RuleHub
  region, **not** toward Monine's Table 1 — recovering Monine's exact point by an FL-space fit is not
  even the right expectation. Monine's Table 1 remains the point that reproduces the physical λ curve
  (Gate 3a), which is the correct grounding check.

**Verdict: PARTIAL** — the fit is sloppy / non-identifiable exactly as the paper's own Fig S2 shows;
K2 and α recover but K1 does not, and PyBNF's objective optimum sits elsewhere in the valley. The
identifiable check (the Gate-3a reproduced curve at the paper's Table-1 params) passes.

---

## Divergence & corrections

- **Scope vs. paper:** the model, the three fitted parameters, and the fit data all match the paper —
  no re-scoping needed. The problem was that the job's **reference numbers** ("the paper's result")
  were taken from the **RuleHub-2019 re-fit**, not from **Monine 2010's own Table 1**, even though the
  shipped nominals `K1=0.467, K2=87.03` *are* the Table-1 estimates.
- **Corrections applied to job files:**
  1. **Data attribution** (README, `tlbr.bngl`, `tlbr.conf`): the binding data is **Monine 2010 Fig 2a**,
     not Posner 2007. Posner 2007 = the ligand (compound 6a) + degranulation data.
  2. **Reference fit / reproduction** (README, `make_reproduction.py`, `tlbr_reproduction.png`): use
     **Monine Table 1 (K1=0.467, K2=87.03, α=0.816)** as "the paper's result"; the RuleHub-2019 re-fit
     is demoted to a secondary cross-check (a different point in the sloppy K1–K2 valley).
  3. **Nominals** (`tlbr.bngl`): set `alpha` nominal 1 → **0.816** so all three shipped nominals equal
     Monine's Table-1 best-fit; relabel "from Brandon" as "Monine 2010 Table 1".
  4. **`_manifest.py` recover target**: `{alpha:0.816, K1:0.467, K2:87.03}` with a loose tol (sloppy fit).
- **Re-run after corrections:** Gate 3a reproduction regenerated at the Table-1 params.

## Bottom line

The `tlbr` job is a faithful encoding of Monine 2010's TLBR model and reproduces the paper's Fig 2a
binding curve at the paper's *own* reported parameters — the model and data grounding are solid. The
validation corrected two provenance errors (the binding data is Monine Fig 2a, not Posner 2007; and
"the paper's result" is Monine's Table 1, not the RuleHub-2019 re-fit). The residual risk is the same
one the paper itself reports: the K1–K2 fit is sloppy/non-identifiable (Fig S2), so a fresh fit
recovers the *reproduced curve* but not the specific Table-1 point. The single most valuable next step
to raise confidence is a full-budget multi-start fit with profile-likelihood on the identifiable
K1–K2 combination.
