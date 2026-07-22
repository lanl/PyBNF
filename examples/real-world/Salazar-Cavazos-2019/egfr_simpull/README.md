# egfr_simpull — multisite EGFR phosphorylation, CHO EGFR-GFP cells, ODE (PyBNF edition-2 job)

A PyBNF edition-2, parameter-fitting job setup derived from:

> Salazar-Cavazos E, Franco Nitta C, Mitra ED, Wilson BS, Lidke KA, Hlavacek WS, Lidke DS.
> **"Multisite EGFR phosphorylation is regulated by adaptor protein abundances and dimer
> lifetimes."** *Mol Biol Cell* 2020; **31**(7):695–708.
> PMCID: [PMC7202077](https://pmc.ncbi.nlm.nih.gov/articles/PMC7202077/) ·
> DOI: [10.1091/mbc.E19-09-0548](https://doi.org/10.1091/mbc.E19-09-0548)

Built with the `curate-pybnf-job` skill. **The authors shipped their own PyBioNetFit setup** —
model, both data files, and the `de`/`chi_sq` config — in supplement
`mc-e19-09-0548-s03/PyBNF-fitting-setup/`. This job is a faithful port of that setup from legacy
BioNetFit to the edition-2 / PEtab surface, so (like `Erickson-2019/igf1r`) the fit recipe is
fully known, not reconstructed.

## The model

EGF reversibly binds EGFR; two liganded receptors **dimerize** (represented as a receptor state
change `II~u`→`II~b`, one kinase becoming the activator `Kin~act` and the other the receiver
`Kin~rec`); the activated dimer *trans*-phosphorylates three C-tail tyrosine classes — **Y1068**,
**Y1173**, and a lumped **YN** ("other sites"); constitutive phosphatases dephosphorylate; **Grb2**
binds pY1068 and **Shc1** binds pY1173. Binding/dimerization/kinase constants are fixed from the
literature (Morimatsu 2007, Low-Nam 2011, Reddy 2016, Kleiman 2011, Kovacs 2015; see the model
header); the six fitted parameters are the two adaptor copy numbers and four phospho-turnover
constants. Deterministic ODE, **75 species, ~1 s to generate — not heavy**, no network cap needed
(dimerization is a state change, not an unbounded aggregate).

## What is fit

Both SiMPull datasets, jointly, in one job (the authors' `fit_v1_28.conf` bound both `.exp`):

| dataset | observable(s) | design | source |
|---|---|---|---|
| % phosphorylation vs. EGF dose | `pY1068_percent()`, `pY1173_percent()`, `phosR_per()` | dose-response scan over `EGFconc` (0.5, 5, 50 nM), each to t_end = 300 s | `dose_resp.exp` |
| % phosphorylation vs. time (25 nM EGF) | same three | time course, 60/120/300 s | `EGF_25nM.exp` |

`pY1068_percent()`/`pY1173_percent()` are the % of EGFR phosphorylated as read by anti-pY1068 /
anti-pY1173; `phosR_per()` is the pan-phosphotyrosine (PY) readout (% phosphorylated at ≥1 site).
All three are model **functions** → parens in the `.exp` header; the `_SD` columns switch the
objective to **`chi_sq`**. (Listed without parens in `_manifest.py`.)

## Files

| file | role |
|---|---|
| `egfr_simpull.bngl` | edition-2, fitting-ready ODE model; six `__FREE` placeholders declared as real ids with best-fit nominals; no actions block |
| `egfr_simpull.conf` | the edition-2 job setup (dose-response parameter scan + 25 nM time course) |
| `dose_resp.exp`, `EGF_25nM.exp` | fit targets (with SD), the authors' own SiMPull data files |
| `make_reproduction.py` | regenerates the reproduction PNG (model at the best-fit vs. data) |
| `egfr_simpull_reproduction.png` | verification figure |
| `make_petab.py` | generates and validates a PEtab bundle from `egfr_simpull.conf` on demand |
| `VALIDATION.md` | provenance / model-fidelity / reproduction record + earned confidence |

## Free parameters (6) — nominals (= published best-fit) vs. an independent PyBNF fit

Nominals in the `.bngl` are the authors' PyBNF best fit (`190127_CHO_EGFR_best-fit.bngl`); the conf
brackets each on a log scale over the authors' `fit_v1_28.conf` bounds. An independent PyBNF
`de`+`refine` fit (population 24, 30 iterations) from those bounds reached **chi_sq = 361.3**,
essentially equal to the published params' **359.7** — i.e. these nominals sit at the global
optimum.

| id | published (best-fit) | independent fit | bounds (log) | recovered? | role |
|---|---|---|---|---|---|
| `GRB2_total_0` | 169853 | 155127 | 1e4–1e6 | ✅ 9 % | total Grb2 copy number (molecules/cell) |
| `SHC1_total_0` | 649426 | 555326 | 1e4–1e6 | ✅ 14 % | total Shc1 copy number (molecules/cell) |
| `ratio_kpkd_Y1068` | 0.15755 | 0.18113 | 0.01–100 | ✅ 15 % | kphos/kdephos ratio at Y1068 |
| `ratio_kpkd_YN` | 0.44476 | 0.44717 | 0.01–100 | ✅ 0.5 % | kphos/kdephos ratio at YN |
| `kdephosYN_0` | 0.017182 | 0.014143 | 0.001–100 | ✅ 18 % | base dephosphorylation rate, YN (/s) |
| `kdephosY1068_0` | 1.6588 | 18.99 | 0.1–100 | ⚠️ sloppy | base dephosphorylation rate, Y1068 (/s) |

Five of six recover within ~18 %. `kdephosY1068_0` is **non-identifiable** (a "fast-rate"
direction): once dephosphorylation is fast relative to the ~300 s protocol the site reaches
quasi-equilibrium and only the **ratio** `ratio_kpkd_Y1068` (which *is* recovered) sets the plateau.
The identifiable quantities — the two adaptor copy numbers and the phospho ratios — are exactly the
paper's biological conclusion (phosphorylation levels track adaptor abundances and dimer lifetimes).

## ✅ PEtab-v2-exportable · not heavy

`scripts/petab_roundtrip.py --job-type de --method ode` exports → lints clean → re-imports (the
observables are arithmetic model functions and the objective is `chi_sq` — inside the exportable
subset; no `normalization`). The network generates in ~1 s, so a full fit runs on a workstation.

The runnable `egfr_simpull.conf` is the single source of truth. `make_petab.py` generates a linted
PEtab bundle on demand; generated copies are deliberately not committed in this gallery.

## Verification (see `VALIDATION.md`)

- **Tier-1** (`scripts/check_conf.py`): edition 2, `job_type=de` resolves, both experiments bind
  data, 6 free params bind by id, no `__FREE`. **PASS.**
- **PEtab round-trip** (`--job-type de --method ode`): export → PEtab v2 lint clean → re-import.
  **PASS.**
- **Real bngsim fit:** reaches a finite objective; the `de`+`refine` fit converges to
  **chi_sq = 361.3 ≈ the published 359.7** (global optimum).
- **Reproduction** (`egfr_simpull_reproduction.png`, model at the published best-fit): total
  **chi_sq = 359.7**, **median relative error 10.4 %** over the 18 fit points. pY1173 is fit tightly
  (≈7 %); the residuals are pY1068 dose saturation (model plateaus ~13 %, data ~20 % at 50 nM) and
  mid-dose pan-PY (model saturates faster than the SiMPull curve) — genuine model–data tension the
  6-parameter fit cannot remove, confirmed by the independent fit landing at the same objective.

## Run

```bash
export BNGPATH="$HOME/Simulations/BioNetGen-2.9.3"   # folder with BNG2.pl
cd examples/real-world/Salazar-Cavazos-2019/egfr_simpull
pybnf -c egfr_simpull.conf                # raise population_size/max_iterations for a full fit
python make_petab.py                      # generate and validate a local PEtab bundle
```

## `_manifest.py` entry (if promoted to the PyBNF real-world corpus)

```python
RealWorldExample(
    folder='Salazar-Cavazos-2019/egfr_simpull', conf='egfr_simpull.conf', simulator='ode',
    observables=('pY1068_percent', 'pY1173_percent', 'phosR_per'),
    system='Multisite EGFR phosphorylation, CHO EGFR-GFP cells (Salazar-Cavazos 2020, '
           'PMC7202077); ODE, SiMPull dose-response + 25 nM time course, chi_sq; '
           'PEtab-exportable. Authors\' own PyBioNetFit setup, ported to edition 2.'),
# The published best-fit is the global optimum (independent fit chi_sq 361.3 vs 359.7); 5/6 params
# recover within ~18%, kdephosY1068_0 is sloppy (fast-rate direction). recover left empty because
# the tier-2 2-iteration fit is too short to converge; see README for the nominal-vs-fit table.
```
