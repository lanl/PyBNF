# tlbr — trivalent-ligand / bivalent-receptor aggregation, network-free NFsim (PyBNF edition-2 job)

A PyBNF edition-2, parameter-fitting job setup. The **network-free model, the fit data, and the
fitted parameters** all come from one paper:

> Monine MI, Posner RG, Savage PB, Faeder JR, Hlavacek WS.
> **"Modeling multivalent ligand-receptor interactions with steric constraints on
> configurations of cell-surface receptor aggregates."**
> *Biophys J* 2010; **98**(1):48–56.
> PMID: [20074516](https://pubmed.ncbi.nlm.nih.gov/20074516/) ·
> DOI: [10.1016/j.bpj.2009.09.043](https://doi.org/10.1016/j.bpj.2009.09.043)

`tlbr.exp` is Monine 2010 **Fig 2a** — a flow-cytometric titration of Alexa-488-labeled trivalent
antigen (compound 6a) binding to IgE-FcεRI on RBL cells; the fitted `K1/K2/α` are Monine 2010
**Table 1** (TLBR column). The trivalent ligand itself (compound 6a) was synthesized, and its
**degranulation** response measured, by:

> Posner RG, Geng D, Haymore S, Bogert J, Pecht I, Licht A, Savage PB.
> **"Trivalent antigens for degranulation of mast cells."**
> *Org Lett* 2007; **9**(18):3551–3554.
> PMID: [17691814](https://pubmed.ncbi.nlm.nih.gov/17691814/) ·
> DOI: [10.1021/ol071231z](https://doi.org/10.1021/ol071231z)

> ⚠️ Posner 2007 reports **degranulation** (β-hexosaminidase, its Fig 3), **not** the binding
> titration fit here. The binding data are Monine 2010's own (Fig 2a). See `VALIDATION.md`.

This is the same fitting problem as **BioNetFit 1 example 3** (Thomas et al. 2016) and PyBNF
2019 paper corpus problem **11-TLBR** (Mitra et al., *iScience* 2019), re-expressed on the
edition-2 surface.

## The model

A **trivalent ligand** `L(s,s,s)` and a **bivalent receptor** `R(s,s)`. Free ligand binds a
receptor site with association strength `K1`; a receptor-bound ligand arm crosslinks a second
receptor with strength `K2`; both bonds break at rate `koff`. Because each ligand carries three
arms and each receptor two sites, crosslinking builds **aggregates that grow without bound** —
there is no finite reaction network — so the model is simulated **network-free with NFsim**
(`method: nf`), with `complex` tracking and a raised global molecule limit (`gml`). Cell-fraction
units (a subvolume `f = 0.001` of a cell; ~300 receptors, up to ~1.3×10⁵ ligands at the top
dose). Stochastic.

## What is fit

One dose-response dataset — the **bound-ligand fraction FL** across a titration of total ligand:

| dataset | observable | design | source |
|---|---|---|---|
| FL vs. total ligand LTconc (12 doses, 5×10⁻⁴–213 nM) | `FL()` = `lambda`/`alpha` = (L_total−L_free)/(2·R_total·alpha) | NF dose-response scan over `LTconc`, each dose to t_end=5000 s | **Monine 2010 Fig 2a** (FL1 = scaled fluorescence; λ = α·FL1, SI Eq 10–11); same 12-dose set as BioNetFit 1 ex 3 |

`FL()` is a model **function** (parenthesized in the `.exp` header). The `.exp` carries **no
`_SD`** columns → the **`sos`** objective.

## Files

| file | role |
|---|---|
| `tlbr.bngl` | edition-2, fitting-ready network-free model (no actions block — NF needs no `generate_network`) |
| `tlbr.conf` | the edition-2 job setup (`method: nf` dose-response scan, `sos`) |
| `tlbr.exp` | fit target — FL() vs. LTconc, 12 doses (no SD) |
| `make_reproduction.py` | regenerates the reproduction PNG (NFsim `parameter_scan` at Monine's Table-1 fit vs. Fig 2a data) |
| `tlbr_reproduction.png` | verification figure |
| `make_petab.py` | generates and validates a PEtab bundle from `tlbr.conf` on demand |
| `VALIDATION.md` | primary-source validation scorecard (5 gates + earned confidence) |

## ✅ The shipped nominals ARE the paper's reported fit (Monine 2010 Table 1)

The shipped nominals **are Monine 2010's own reported best-fit** for the TLBR model (Table 1) — the
model runs, as-is, at the paper's parameters and reproduces Fig 2a (RMS in λ = 0.015 < the paper's
own RMS<0.02 acceptance, SI Eq 11):

| id | **Monine 2010 Table 1 (TLBR)** | 68 % CI | role |
|---|---|---|---|
| `alpha` | **0.816** | 0.758–0.881 | overall scale of the FL readout (λ = α·FL1) |
| `K1` | **0.467** | 0.111–0.767 | free-ligand / receptor-site equilibrium constant (/nM) |
| `K2` | **87.03** | 31.6–128.1 | crosslinking equilibrium constant (/nM) |

> A previous version of this job treated these as a mere "starting point" and used a **different**
> fit — the PyBNF/Mitra-2019 re-fit (RuleHub
> [`11-TLBR/fit_ss`](https://github.com/RuleWorld/RuleHub/tree/2019Aug21/Published/Mitra2019/11-TLBR/fit_ss)
> init7: `α=0.746, K1=0.109, K2=33.6`, `sos=0.00214`). That re-fit lands at a **different point of
> the same sloppy K1–K2 valley** (Monine's Fig S2): it reproduces the *normalized* FL1 but, because
> its α differs, undershoots Fig 2a's λ curve — it is **not** the paper's reported result.

**The fit is sloppy / non-identifiable** — as the paper's own Fig S2 shows and `VALIDATION.md`
Gate 3b confirms: many `(K1, K2, α)` triples fit the data comparably (a budget PyBNF fit recovers
`K2` within ~4 % and `α` within tolerance, but `K1` only to ~3×). The identifiable check is the
**reproduced curve**, not the individual constants. See the paper-level [`../README.md`](../README.md)
and `VALIDATION.md`.

## Changes from the source model (documented in `tlbr.bngl`)

- **No `begin actions` block** — a network-free model needs no `generate_network`; the
  dose-response scan is synthesized from `tlbr.conf`. (This is the correct NF shape — contrast
  the network-generating `Kozer-2013/egfr_ode`, whose configuration declares a finite-network
  cap.)
- Fitted constants (`alpha`, `K1`, `K2`) bind to model ids **by name** (ADR-0034, no `__FREE`).
  Otherwise the model network is byte-for-byte the Monine 2010 / BioNetFit-1 model.

## ✅ PEtab-v2-exportable · ✅ workstation-runnable

Verified with `scripts/petab_roundtrip.py --job-type de --method nf` (export → lint clean →
import). Unlike the heavy `egfr_nf`, a single dose-response scan here is **fast** (~11 s for all
12 doses via one `parameter_scan`), so the reproduction runs on a workstation in ~1 minute; a
full metaheuristic fit (population × iterations × doses × replicates) is still best on a cluster.

The runnable `tlbr.conf` is the single source of truth. `make_petab.py` generates a linted PEtab
bundle on demand; generated copies are deliberately not committed in this gallery.

## Verification

- **Tier-1** (`scripts/check_conf.py`): edition 2, `job_type=de` resolves, the experiment binds
  data, `model.stochastic = True` (→ `simulator='nf'`), 3 free params bind by id, no `__FREE`.
  **PASS.**
- **PEtab round-trip** (`--job-type de --method nf`): export → PEtab v2 lint clean → re-import.
  **PASS.**
- **Reproduction** (`tlbr_reproduction.png`, NFsim `parameter_scan` at **Monine's Table-1 fit**,
  7 replicates averaged): reproduces the Monine 2010 Fig 2a titration to **median 2.4 % / max 11 %
  relative error** (over the 9 doses with FL > 0.05; the 3 lowest doses sit at the near-zero noise
  floor, one datum even negative), **sos ≈ 0.0038** (FL space) and **RMS ≈ 0.015 in λ**, inside
  Monine's own RMS<0.02 acceptance (SI Eq 11). **PASS.**
- **Primary-source validation** (`VALIDATION.md`): 5 gates against Monine 2010 + SI + Posner 2007;
  model identical to the published TLBR model, data traced to Fig 2a, reproduction at the paper's
  Table-1 params — **earned confidence 91/100** (the residual is the paper's own fit
  non-identifiability + an overlay-not-original-file data match, not any job defect).

## Run

```bash
export BNGPATH="$HOME/Simulations/BioNetGen-2.9.3"   # folder with BNG2.pl
cd examples/real-world/Monine-2010/tlbr
pybnf -c tlbr.conf                    # raise population_size for a full fit (the fit is sloppy; see VALIDATION.md)
python make_petab.py                 # generate and validate a local PEtab bundle
```

## `_manifest.py` entry (if promoted to the PyBNF real-world corpus)

```python
RealWorldExample(
    folder='Monine-2010/tlbr', conf='tlbr.conf', simulator='nf', stochastic=True,
    observables=('FL',),
    system='Trivalent-ligand/bivalent-receptor aggregation (Monine 2010, PMID 20074516; model, '
           'Fig 2a binding data, and Table 1 fit; ligand from Posner 2007, PMID 17691814; '
           'BioNetFit 1 ex 3); network-free NFsim, dose-response, sos; PEtab-exportable',
    recover={'alpha': 0.816, 'K1': 0.467, 'K2': 87.03}, tol=1.0),
    # Monine 2010 Table 1 (TLBR): the paper's OWN reported best-fit. The K1-K2 fit is sloppy /
    # non-identifiable (Monine Fig S2; VALIDATION.md Gate 3b) -- loose tol; K2 and alpha recover
    # well, K1 only to ~3x. (RuleHub Mitra-2019 re-fit alpha=0.746,K1=0.109,K2=33.6 is a DIFFERENT
    # valley point that minimizes the FL-space sos, not Monine's lambda-space RMS.)
```
