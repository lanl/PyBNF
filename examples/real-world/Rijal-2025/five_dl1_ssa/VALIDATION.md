# VALIDATION — Rijal-2025/five_dl1_ssa

Primary-source and runtime validation of the exact-SSA 5DL1 job.

> **Confidence: 70 / 100.** Data provenance and the four-reaction stochastic model pass, and the
> complete edition-2 SSA fit path runs with finite values. Exact SSA also agrees with analytical
> stationary moments. The lower score reflects a material result-level limitation: under the
> necessary steady-state elimination of unpublished row-specific `kon_R` values, Rijal's Fig. 7
> parameters give 46.28% mean Fano error rather than the paper's reported 28%, and a short fit does
> not recover `gamma`.

## Gate 0 — materials inventory

- Rijal and Mehta 2025 paper — model, Eq. 14, and Fig. 7 parameters; identified by the DOI in
  the job README (the copyrighted PDF is not vendored).
- Jones et al. 2014 paper and supplement — experimental Fig. 3A, methods, and rate context;
  identified by the DOI in the job README (the PDFs are not vendored).
- Authors' numeric data/code: DGA GitHub repository — exact arrays and preprocessing; obtained and
  represented here by the traceable source table plus deterministic conversion script.

**Verdict: PASS.** No author BNGL exists; the paper describes a four-reaction process and the
authors' released implementation supplies the exact numeric fit target.

## Gate 1 — data provenance

The 18 mean/Fano rows in `jones_fig3a_source.tsv` match the authors'
`science_data_5DL1.npy` array to a maximum absolute difference of 4.9e-12. `make_data.py`
applies the released notebook's
`np.unique(mean, return_index=True)` operation, retaining the first of the duplicate
mean=4.8948732583 rows, then computes `SD = sqrt(mean * Fano)`. This produces all 17 rows and all
three numeric columns of `five_dl1.exp` deterministically.

**Verdict: PASS.** The fit data are traceable to Jones Fig. 3A through the later paper's exact
released array, with a transparent, reversible transformation.

## Gate 2 — model fidelity

| aspect | Rijal Fig. 3A | local BNGL | verdict |
|---|---|---|---|
| promoter states | unbound/active and repressor-bound/inactive | `Promoter(state~on~off)` | match |
| switching | `kon_R`, `koff_R` | two unimolecular rules | match |
| expression | active transcription at `r` | active promoter produces `mRNA()` | match |
| decay | mRNA degradation at `gamma` | first-order degradation | match |
| normalization | `koff_R = 1` | fixed at one | match |

The generated network is finite and tiny: three species and four reactions. Exact BNGsim SSA
replaces the paper's DGA sigmoid/Gaussian relaxations, as required by this task. The only fit-design
adaptation is the documented steady-state elimination
`kon_R = r_over_gamma / mean_target - 1`; it changes how per-row inputs are assigned, not the
reaction network.

**Verdict: PASS (dynamics), EQUIVALENT REDUCTION (fit inputs).**

## Gate 3a — reproduction at the paper's parameters

`make_reproduction.py 2000` uses `BngsimModel`, `method=ssa`, and the Rijal Fig. 7 values
`r=87.48`, `gamma=9.80`, `koff_R=1` (`r/gamma=8.92653`). Across 2,000 trajectories per row:

- Rijal Eq. 14 SOS from the exact-SSA moments: **4.34375**;
- mean absolute Fano percentage error against the Jones curve: **46.28%**;
- SSA vs. analytical mean: **5.17% median**, 12.3% maximum relative error;
- SSA vs. analytical SD: **3.43% median**, 16.0% maximum relative error.

The agreement with analytical stationary moments verifies the SSA implementation. The mismatch
with Rijal's reported 28% data error is a fit-design discrepancy, not Monte Carlo failure: the paper
optimized latent per-row `kon_R` values and a transient DGA objective, whereas this reduced job
derives them from the stationary mean. The committed `five_dl1_ssa_reproduction.png` makes that
partial reproduction visible.

**Verdict: PARTIAL.**

## Gate 3b — bounded fit through PyBNF

A deliberately short DE run (`population_size=6`, `max_iterations=2`) using the committed
200-trajectory smoothing completed through the edition-2 BNGsim path in about 8.8 seconds:

| quantity | Rijal Fig. 7 | short-fit result |
|---|---:|---:|
| objective | not directly comparable | 0.662755 |
| `r_over_gamma` | 8.92653 | 8.37541 |
| `gamma` | 9.80 | 5.36682 |
| derived `r` | 87.48 | 44.949 |

The objective falls and `r_over_gamma` is close, but `gamma` is not recovered under this tiny
budget and reduced input parameterization. The important operational result is that parameter scan,
SSA replication, moment materialization, objective evaluation, and optimizer update run end to end.

**Verdict: PARTIAL — operational fit demonstrated; parameter recovery is incomplete.**

## Gate 4 — configuration and interchange

- Curate-skill `check_conf.py`: PASS; edition 2, DE, two bound parameters, stochastic BNGsim job.
- Curate-skill `petab_roundtrip.py`: PASS; export, PEtab-v2 lint, and re-import are clean.
- Semantic caveat: PEtab core does not carry PyBNF's replicate/smoothing controls, so the portable
  bundle alone does not reproduce the SSA ensemble moments.

**Verdict: PASS for PyBNF execution and structural interchange.**

## Bottom line

This job is useful as a small end-to-end exact-SSA fit and as a transparent negative result: the
available publication artifacts are insufficient to recreate the reported 5DL1 fit under the
reduced stationary parameterization. The highest-value next step is to obtain or reconstruct the
17 fitted `kon_R` values and test the complete parameterization with exact SSA.
