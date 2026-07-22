# egfr_ode — EGFR higher-order oligomerisation & phosphorylation, ODE (PyBNF edition-2 job)

A PyBNF edition-2, parameter-fitting job setup derived from:

> Kozer N, Barua D, Orchard S, Nice EC, Burgess AW, Hlavacek WS, Clayton AHA.
> **"Exploring higher-order EGFR oligomerisation and phosphorylation — a combined
> experimental and theoretical approach."** *Mol BioSyst* 2013; **9**(7):1849–1863.
> PMCID: [PMC3698845](https://pmc.ncbi.nlm.nih.gov/articles/PMC3698845/) ·
> DOI: [10.1039/c3mb70073a](https://doi.org/10.1039/c3mb70073a) · PMID: 23629589

Built with the `curate-pybnf-job` skill. This is the **deterministic ODE** (network-generating)
member of the `Kozer-2013` pair; the network-free NFsim sibling (Mitra 2019 reformulation) is
[`../egfr_nf`](../egfr_nf/). Part of the PyBNF 2019 paper corpus (Mitra et al., *iScience* 2019;
BioNetFit 1 problem 2), on the edition-2 surface.

## The model

EGF binds EGFR, and EGFR ectodomains crosslink into higher-order oligomers (capped at
**tetramers**, including **ring-closure** reactions), whose crosslinked cytosolic tails undergo
an activating conformational change and *trans*-phosphorylate. Binding/crosslinking constants
are fixed from the literature (Macdonald & Pike 2008, Elleman 2001, Low-Nam 2011; Table 1 of
Kozer 2013); five rate constants and four observable-scale factors are fit. Deterministic ODE.

## 🔑 Network cap lives in the job configuration

The reaction network is **finite only under a `max_stoich` cap** — the crosslinking + ring rules
do *not* themselves bound oligomer size. The edition-2 model intentionally has no actions block;
`egfr_ode.conf` declares the model-scoped network option instead:

```
generate_network = max_stoich=>{EGF=>4,EGFR=>4}
```

PyBNF synthesizes the capped `generate_network` call together with the time-course and
dose-response actions. Removing the option would fall back to an unbounded network generation
that never terminates; the backend-free test suite guards this configuration path (#473).

## What is fit

Both **preprocessed** Kozer datasets (each scaled so its own measurement average is 1), in one job:

| dataset | observable | design | figure |
|---|---|---|---|
| EGFR cluster density vs. time (30 nM) | `pre2_time` = `alpha2_pre`·Clusters | time course, 0–600 s | Fig. 3B |
| phospho-EGFR vs. time (30 nM) | `pre4_time` = `alpha4_pre`·pEGFR | time course, 0–600 s | Fig. 3D |
| EGFR cluster density vs. EGF dose | `pre1_dose` = `alpha1_pre`·Clusters | dose-response scan over `LT`, to t_end=1200 s | Fig. 2B |
| phospho-EGFR vs. EGF dose | `pre3_dose` = `alpha3_pre`·pEGFR | dose-response scan over `LT`, to t_end=1200 s | Fig. 2D |

Both `.exp` carry `_SD` columns → the **`chi_sq`** objective.

## Files

| file | role |
|---|---|
| `egfr_ode.bngl` | edition-2, fitting-ready ODE model with no actions block |
| `egfr_ode.conf` | the edition-2 job setup, including the network cap and time course + dose-response scan |
| `timecourse.exp`, `doseresponse.exp` | fit targets (with SD), Kozer 2013 preprocessed data |
| `make_reproduction.py` | regenerates the reproduction PNG (model at the PyBNF best-fit vs. data) |
| `egfr_ode_reproduction.png` | verification figure |

## Free parameters (9) — nominals vs. the PyBNF best-fit

The shipped nominals are **Kozer 2013's own Table-1 fit**, which reproduces the *raw* data; our
`.exp` is *average-normalized*, so the reproduction uses the **PyBNF best-fit** (Mitra 2019 Data
S1, RuleHub [`10-egfr/fit_de`](https://github.com/RuleWorld/RuleHub/tree/2019Aug21/Published/Mitra2019/10-egfr),
chi_sq = 10.16):

| id | nominal (Kozer 2013) | **PyBNF best-fit** | scale (conf) | role |
|---|---|---|---|---|
| `k_o` | 6 | **0.20131** | log 0.01–100× | ligand-mediated activating tail change (/s) |
| `k_c` | 1.6 | **0.38559** | log | ligand-independent deactivating tail change (/s) |
| `kaf` | 15.4 | **161.351** | log | EGFR tail-tail association (/s) |
| `kar` | 8.89 | **5.5574** | log | EGFR tail-tail dissociation (/s) |
| `chi_r` | 43700 | **1770.02** | log | ring-closure enhancement factor (nM) |
| `alpha1_pre` | 0.262289257 | **25.6534** | linear 0–200× | cluster-density scale, dose (Fig. 2B) |
| `alpha2_pre` | 0.30214449 | **29.0337** | linear | cluster-density scale, time (Fig. 3B) |
| `alpha3_pre` | 0.414521959 | **39.7741** | linear | phospho-EGFR scale, dose (Fig. 2D) |
| `alpha4_pre` | 0.28729663 | **30.5021** | linear | phospho-EGFR scale, time (Fig. 3D) |

The fit is **sloppy/non-identifiable** (kinetics disagree across the paper's four algorithms at
equal objective); the `alpha*_pre` scale factors are the well-determined part (all algorithms
land ~90–106× the nominal). See the paper-level [`../README.md`](../README.md).

## ⚠️ Heavy (cluster-scale network)

The crosslinking+ring network (even capped at tetramers) takes **~15–20 min** to generate
through BNG2.pl, and the published fit used large populations on a cluster — so a full fit is a
cluster job. The reproduction needs just **one** generation (topology is dose- and
kinetics-independent).

The runnable `egfr_ode.conf` is the single source of truth. Its model-scoped network cap has no
portable PEtab v2 field, so this gallery does not publish a generated PEtab bundle for this job.

## Verification

- **Tier-1** (`scripts/check_conf.py`): edition 2, `job_type=de` resolves, both experiments bind
  data, 9 free params bind by id, no `__FREE`. **PASS.**
- **Reproduction** (`egfr_ode_reproduction.png`, model at the PyBNF best-fit): reproduces the
  four Kozer datasets to **median 5–14 % relative error** (cluster density time 13.3 %, phospho
  time 6.0 %, cluster density dose 5.0 %, phospho dose 13.9 %); summed **chi_sq = 10.16 = the
  published objective (10.164)** — exact, since the ODE reproduction is deterministic. The `max`
  outliers are the near-zero t=0 / lowest-dose phospho points.

## Run

```bash
export BNGPATH="$HOME/Simulations/BioNetGen-2.9.3"   # folder with BNG2.pl
cd examples/real-world/Kozer-2013/egfr_ode
pybnf -c egfr_ode.conf                # cluster-scale: raise population/iterations on a cluster
```

## `_manifest.py` entry (if promoted to the PyBNF real-world corpus)

```python
RealWorldExample(
    folder='Kozer-2013/egfr_ode', conf='egfr_ode.conf', simulator='ode', heavy=True,
    observables=('pre1_dose', 'pre2_time', 'pre3_dose', 'pre4_time'),
    system='EGFR activation & higher-order clustering (Kozer 2013, PMC3698845); ODE '
           '(network-generating, cluster-scale, generate_network+max_stoich in the .conf), '
           'time course + dose-response, chi_sq; native network-cap option'),
    recover={'alpha2_pre': 29.0337, 'alpha4_pre': 30.5021}, tol=0.5),
    # PyBNF best-fit: RuleHub Published/Mitra2019/10-egfr/fit_de (chi_sq 10.16). Kinetics are
    # sloppy/non-identifiable; the alpha*_pre scale factors are the well-determined recover targets.
```
