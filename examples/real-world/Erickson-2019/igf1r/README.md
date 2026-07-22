# igf1r — IGF1 / IGF1R harmonic-oscillator binding fit (Erickson 2019's actual published fit)

A PyBNF **edition-2 (new-era)** parameter-fitting job (`igf1r.conf` / `igf1r.bngl`); the
SI-verbatim **edition-1** twin is kept as `igf1r_legacy.conf` / `igf1r_legacy.bngl` (provenance +
the BNG2.pl reproduction oracle). It reproduces the IGF1-IGF1R binding fit of Erickson et al. 2019
(**Table 1**, **Fig 3**): **seven** rate constants fit **simultaneously to three** Kiselyov-2009
datasets — the steady-state competition curve (Fig 5B) and the 20- and 60-minute dissociation
curves (Fig 5D). The eighth constant (`a2prime`) is fixed by the detailed-balance constraint
(Erickson Eq 1).

> **Now runs in edition-2 (issue [lanl/PyBNF#474](https://github.com/lanl/PyBNF/issues/474)).**
> The two F5D experiments are a stateful **2 h-preincubate → wash → cold-competition-scan** protocol.
> PyBNF edition-2 previously could not express it (no `parameter_scan` after pre-equilibration, no
> species `setConcentration` intervention), so this job was legacy-only. #474 added both — plus the
> bngsim carried-state scan ([lanl/bngsim#11](https://github.com/lanl/bngsim/issues/11),
> bngsim ≥ 0.11.34) — so the whole protocol is now **synthesized from `igf1r.conf`** with **no
> in-model actions block**. The legacy job is retained, clearly labeled, because its model is
> byte-identical to the paper's SI file that produced Table 1.

> **[fitting problem / job source]** Erickson KE, Rukhlenko OS, Shahinuzzaman M, Slavkova KP,
> Kholodenko BN, Hlavacek WS, et al. **"Modeling cell line-specific recruitment of signaling
> proteins to the insulin-like growth factor 1 receptor."**
> *PLoS Comput Biol* 2019; **15**(1):e1006706.
> PMCID: [PMC6353226](https://pmc.ncbi.nlm.nih.gov/articles/PMC6353226/) ·
> DOI: [10.1371/journal.pcbi.1006706](https://doi.org/10.1371/journal.pcbi.1006706)
> *(parameter estimation performed with BioNetFit 1; model + data files in the paper's SI
> "S2 Compressed File Archive")*
>
> **[model + data origin]** Kiselyov VV, Versteyhe S, Gauguin L, De Meyts P.
> **"Harmonic oscillator model of the insulin and IGF1 receptors' allosteric binding and
> activation."** *Mol Syst Biol* 2009; **5**:243.
> PMID: [19225456](https://pubmed.ncbi.nlm.nih.gov/19225456/) ·
> DOI: [10.1038/msb.2008.78](https://doi.org/10.1038/msb.2008.78)
> (Fig 5B = competition; Fig 5D = 20/60-min dissociation — the F5B / F5D_20min / F5D_60min data.)

> **Re-scoped 2026-07-11 to match the literature.** This slug previously carried a reduced
> 3-parameter (`K1`/`K2`/`K1prime`) fit of **F5B only** — a PyBNF teaching distillation that is
> **not** in Erickson's paper. The paper's SI ships the authors' own BioNetFit model+conf (the
> exact files that produced Table 1); this job is now built from those, so the model, the three
> datasets, the seven free constants, the detailed-balance constraint, and the objective all match
> what was published. See [`VALIDATION.md`](VALIDATION.md).

## The biochemistry

A **radioligand competition / dissociation assay** on the IGF1 receptor. IGF1R is a pre-formed,
disulfide-bonded **dimer** with two ligand sites (Site 1, Site 2); a single IGF1 can engage a site
and **crosslink** the two sites *within* the dimer (avidity), following the Kiselyov harmonic-
oscillator mechanism. Labelled ("hot") IGF1 is held fixed while unlabelled ("cold") IGF1 is
titrated; the readout is hot-ligand binding vs. cold dose. Deterministic **ODE**; **finite network
without a cap** (crosslinking is confined to the dimer, so the bare `generate_network({overwrite=>1})`
terminates).

## What is fit

| dataset | design | observable → data | source |
|---|---|---|---|
| `F5B` | 4 h steady-state competition at **7 pM** hot IGF1; parameter scan over cold dose (18 doses) | `IGF1_hot_bound`, normalized to no-competitor | Kiselyov 2009 **Fig 5B** |
| `F5D_20min` | **2 h preincubation at 24 pM** hot, wash (free hot→0), then **20 min** cold competition; scan over cold (10 doses) | `IGF1_hot_bound`, % remaining | Kiselyov 2009 **Fig 5D** (20 min) |
| `F5D_60min` | same protocol, **60 min** competition | `IGF1_hot_bound`, % remaining | Kiselyov 2009 **Fig 5D** (60 min) |

All three fit **jointly**. Seven free rate constants (`a1_perMpers`, `d1`, `a2_perMpers`, `d2`,
`kcr`=a′₁, `d1prime`=d′₁, `d2prime`=d′₂); `a1prime=kcr` and `a2prime` is set by detailed balance
`a2prime = (a2·a1prime·d1·d2prime)/(a1·d1prime·d2)`. Objective **`chi_sq`** (weighted LS by the
per-point `_SD`), **`normalization=init`**.

## Free parameters — Kiselyov nominal vs. Erickson Table 1 (the fit target)

The model ships `__FREE` placeholders (the fit estimates them); `make_reproduction.py` sets them to
the Erickson Table-1 values below. The "Kiselyov" column is the earlier estimate the paper compares
against (Table 1, right column).

| token (paper symbol) | Kiselyov 2009 | **Erickson 2019 Table 1** (best-fit) | 95% CI (Table 1) |
|---|---|---|---|
| `a1_perMpers` (a₁, M⁻¹s⁻¹) | 3.5×10⁵ | **2.8×10⁵** | 2.3×10⁵ – 9.7×10⁵ |
| `d1` (d₁, s⁻¹) | 3.2×10⁻³ | **5.0×10⁻²** | 1.1×10⁻² – 0.13 |
| `a2_perMpers` (a₂, M⁻¹s⁻¹) | 8.7×10³ | **1.5×10⁴** | 3.5×10³ – 2.2×10⁴ |
| `d2` (d₂, s⁻¹) | 4.2×10⁻³ | **1.9×10⁻⁴** | 1.1×10⁻⁵ – 2.6×10⁻⁴ |
| `kcr` (a′₁, s⁻¹) | 0.33 | **5.6×10⁻³** | 1.6×10⁻⁴ – 0.29 |
| `d1prime` (d′₁, s⁻¹) | =d₁ | **1.9×10⁻⁵** | 3.8×10⁻⁷ – 6.9×10⁻⁴ |
| `d2prime` (d′₂, s⁻¹) | =d₂ | **1.3×10⁻²** | 3.8×10⁻³ – 2.7×10⁻² |
| `a2prime` (a′₂, s⁻¹) — *derived* | =a′₁ | **52** (detailed balance) | 0.45 – 5.7×10⁴ |

From Table 1, site KDs are d₁/a₁ = 179 nM and d₂/a₂ = 13 nM ("13 nM and 180 nM" in the paper).

## Files

| file | role |
|---|---|
| `igf1r.bngl` | **edition-2 model** — 7 fit constants bound by id (ADR-0034), detailed balance, **no actions block** (the protocol is synthesized from `igf1r.conf`) |
| `igf1r.conf` | **edition-2 job** (`edition = 2`, `experiment:`/`condition:` surface, `chi_sq`, `normalization=init`, `ss`+refine) — the primary job |
| `igf1r_legacy.bngl` | Erickson's SI binding model **verbatim** (7 `__FREE` tokens + detailed balance + the full incubate/wash/scan actions block) — provenance |
| `igf1r_legacy.conf` | edition-1 twin (`model = igf1r_legacy.bngl : F5B.exp, …`) — the BNG2.pl reproduction path |
| `F5B.exp` · `F5D_20min.exp` · `F5D_60min.exp` | the three fit targets (Kiselyov Fig 5B/5D; per-point `_SD`) |
| `make_reproduction.py` | reproduces Erickson Fig 3A/3B at the Table-1 params (drives `igf1r_legacy.bngl` through BNG2.pl) |
| `igf1r_reproduction.png` | reproduction figure |
| `VALIDATION.md` | primary-source validation scorecard (five gates + earned confidence) |

## How edition-2 expresses the F5D protocol (issue #474)

The two F5D experiments are a stateful **2 h-preincubate → wash → cold-competition-scan** protocol.
Edition-2 synthesizes it from `igf1r.conf` — no in-model actions block:

- **`preequilibrate: incubate` + `equil_t_end: 7200`** — the fixed-time pre-incubation at 24 pM hot
  (`condition: incubate` sets the hot species amount);
- **`condition: wash`** — a **species `setConcentration` intervention**: `"IGF1(ds,hs,label~hot)" = 0`
  (zero free hot; bound hot remains — what dissociates) and `"IGF1(ds,hs,label~cold)" =
  IGF1_cold_conc*(NA*Vecf)` (the competitor, re-asserted so it tracks the scanned dose);
- **`type: parameter_scan`** — the measured dose-response, each dose reset to the carried post-wash
  state (bngsim's native carried-state scan, [lanl/bngsim#11](https://github.com/lanl/bngsim/issues/11),
  bngsim ≥ 0.11.34).

F5B is a plain dose-response scan at the model-default 7 pM hot. See
[lanl/PyBNF#474](https://github.com/lanl/PyBNF/issues/474) for the capability that made this possible.

## ⚠️ NATIVE-ONLY (not PEtab-exportable, for now)

`normalization=init` (a whole-fit PyBNF prediction transform with no PEtab v2 operator) makes this
job native-only regardless. Separately, PEtab **export** of the edition-2 preincubate→wash→dose-scan
shape (a species-amount condition + a pre-equilibrated dose-response) is **deferred** — a follow-up
to ADR-0052's phased export; the fitter supports it, the exporter refuses it with a clear message.

## Verification (see VALIDATION.md for the full scorecard)

- **Gate 1 (data):** `F5B.exp` is byte-identical to the SI copy; all three `.exp` are the authors'
  own extractions of Kiselyov Fig 5B/5D (confirmed against the rendered figure).
- **Gate 2 (model):** `igf1r_legacy.bngl` is **byte-identical** to the SI `IGF1R_fit.bngl` (the file
  that produced Table 1), modulo the `__FREE__`→`__FREE` token spelling; generated network 27 species
  / 96 reactions. The edition-2 `igf1r.bngl` is the same model block with the 7 tokens bound by id
  and the actions block removed (synthesized from `igf1r.conf`).
- **Gate 2b (edition-2 = legacy):** the edition-2 job, run through the bngsim backend at the Table-1
  params, reproduces the paper to the same tolerance as the BNG2.pl legacy path — **F5B 1.0 %,
  F5D_20min 5.3 %, F5D_60min 7.0 %** median rel err — i.e. the synthesized preincubate→wash→dose-scan
  protocol matches the hand-written actions block (issue lanl/PyBNF#474).
- **Gate 3a (reproduce Fig 3 at Table-1 params):** `make_reproduction.py` overlays the model at
  Erickson's Table-1 values on all three datasets — **F5B median 1.1 % rel err**, **F5D_20min 6.2 %**,
  **F5D_60min 6.5 %** (see `igf1r_reproduction.png`); reproduces Fig 3A/3B.
- **Gate 3b (recover Table 1 by fitting):** a scatter-search run converges to **Obj 6.74 ≤ the
  Table-1 objective (7.65)** — **6/7 constants within 3×** and **7/7 within the paper's own 95 % CIs**
  (the one 3× miss is `d1prime`, a crosslinking constant the paper reports as non-identifiable). See
  VALIDATION.md.

## Run

```bash
export BNGPATH="$HOME/Simulations/BioNetGen-2.9.3"   # folder with BNG2.pl (network generation)
cd examples/real-world/Erickson-2019/igf1r

pybnf -c igf1r.conf            # edition-2 ODE fit (needs bngsim >= 0.11.34; ~4 scans per evaluation)
pybnf -c igf1r_legacy.conf     # edition-1 twin (BNG2.pl backend; the SI-verbatim actions block)
python make_reproduction.py    # reproduction figure at the Erickson Table-1 params (BNG2.pl)
```

The edition-2 fit runs on the **bngsim** backend (`edition >= 2 ⇒ bngsim`); BNG2.pl is still used
once to expand the rules into a reaction network. The carried-state F5D dose-scan needs
**bngsim ≥ 0.11.34** (lanl/bngsim#11).

## `_manifest.py` entry (if promoted to the PyBNF real-world corpus)

```python
RealWorldExample(
    folder='Erickson-2019/igf1r', conf='igf1r.conf', simulator='ode',
    observables=('IGF1_hot_bound',),
    system='IGF1/IGF1R harmonic-oscillator binding (Erickson 2019 fit, PMC6353226, Table 1/Fig 3; '
           'Kiselyov 2009 model+data, PMID 19225456, Fig 5B/5D); EDITION-2 job: 7 free rate '
           'constants + detailed balance, 3 datasets (F5B+F5D_20min+F5D_60min), preincubate->wash->'
           'dose-scan protocol (parameter_scan + pre-equilibration + species setConcentration, '
           '#474), chi_sq, normalization=init -> NATIVE-ONLY (normalization + deferred export)'),
    # Edition-2 (no in-model actions; needs bngsim >= 0.11.34 for the carried-state F5D scan).
    # Sloppy fit (site KDs / apparent affinity identifiable; individual constants less so) ->
    # a recover assertion, if any, needs a loose tol. Reproduction at Table-1: F5B ~1% rel err.
```
