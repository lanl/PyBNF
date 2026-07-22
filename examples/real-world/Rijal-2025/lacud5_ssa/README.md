# lacud5_ssa — exact-SSA fit to the Jones lacUV5/lacUD5 noise curve

This is a small PyBNF edition-2 stochastic fit of the two-state promoter in Rijal and Mehta
(2025), Fig. 3A, to the lacUV5 mRNA mean/Fano measurements in Jones et al. (2014), Fig. 3A.
Rijal's paper and released data call the promoter `lacUD5`; Jones calls it `lacUV5`. The job uses
the Rijal spelling while preserving the Jones provenance.

## What is fit

The BNGL model has an active and an inactive promoter, transcription from the active state, and
first-order mRNA degradation. `koff_R` is fixed to one. Differential evolution fits:

| parameter | role | Rijal Fig. 7 nominal | bound |
|---|---|---:|---:|
| `r_over_gamma` | unrepressed mean capacity | 90.25 / 6.20 = 14.55645 | 10.05–25 |
| `gamma` | mRNA degradation rate | 6.20 | 1–25, log-uniform |

The transcription rate is `r = r_over_gamma * gamma`. Each of the 17 scan rows obtains its
repressor association rate from
`kon_R = r_over_gamma / mean_target - 1`, with `koff_R = 1`.

Every objective evaluation runs 200 exact BNGsim SSA trajectories. The model emits terminal
`mRNA_count` and `mRNA_squared`; after replicate smoothing the measurement layer evaluates
`Mean_mRNA = E[m]` and `mRNA_SD = sqrt(E[m^2] - E[m]^2)`. Consequently `objective = sos`
implements the mean-plus-standard-deviation loss in Rijal Eq. 14.

## Data provenance

`jones_fig3a_source.tsv` is a 12-significant-digit text transcription of the 18-row
`science_data_lacUD5.npy`
array released in the
[DGA authors' repository][dga].
That array digitizes Jones Fig. 3A. `make_data.py` reproduces the authors' preprocessing:

1. retain the first row at each unique mean, leaving 17 rows;
2. convert Fano factor to mRNA SD with `SD = sqrt(mean * Fano)`;
3. write `mean_target`, `Mean_mRNA`, and `mRNA_SD` to `lacud5.exp`.

The mean is used both as the experimental observable and as the scan input needed to eliminate
the unpublished row-specific `kon_R`. This is an explicit reduced steady-state parameterization;
it is not an independent dose variable.

## Files

| file | role |
|---|---|
| `lacud5_ssa.bngl` | four-reaction two-state promoter model |
| `lacud5_ssa.conf` | edition-2 DE job, `method: ssa`, 200-replicate smoothing |
| `jones_fig3a_source.tsv` | 18 source mean/Fano pairs |
| `lacud5.exp` | 17-row fitting table |
| `make_data.py` | deterministic source-to-fit-data conversion |
| `make_reproduction.py` | 2,000-trajectory BNGsim SSA verification at published parameters |
| `lacud5_ssa_reproduction.png` | committed verification figure |
| `VALIDATION.md` | primary-source and runtime validation |

## Run

```bash
export BNGPATH=/path/to/BioNetGen
pybnf -c lacud5_ssa.conf
python make_reproduction.py       # optional argument: trajectory count
```

For reproducible validation on systems where BNGsim cannot write its code-generation cache, set
`BNGSIM_NO_CODEGEN=1`; this changes generated-propensity caching, not the SSA method.

## Scope and limitations

Rijal's fit optimized 17 independent `kon_R` values and used a transient differentiable-Gillespie
surrogate. Those fitted inputs are unavailable. This job instead uses the exact steady-state mean
identity and exact SSA at `t_end=10`, the horizon of the paper's exact two-state benchmarks. It is
therefore a practical real-data SSA fitting example and PyBNF's workstation-scale end-to-end SSA
regression for issue #472, not a claim to reproduce the paper's complete optimizer or
nuisance-parameter fit.

PEtab-v2 export/lint/import is structurally clean, but SSA replication and smoothing are PyBNF
execution controls outside the portable PEtab core.

## Repository validation

This job is registered in [`../../_manifest.py`](../../_manifest.py). Default CI checks its
edition-2 configuration and data bindings without a simulator; the opt-in `recovery` tier drives
a bounded fit through the real BNGsim SSA backend.

[dga]: https://github.com/Emergent-Behaviors-in-Biology/Differentiable-Gillespie-Algorithm
