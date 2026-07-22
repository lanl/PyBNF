# Rijal-2025 — exact-SSA fits of two-state promoter noise

Four compact PyBNF edition-2 jobs derived from Rijal and Mehta's two-state promoter model and
the experimental *E. coli* mRNA-noise data of Jones et al. The two exact-SSA jobs provide the
small, real-world, end-to-end stochastic fits requested by
[PyBNF issue #472](https://github.com/lanl/PyBNF/issues/472), using
`experiment: ... method: ssa` through BNGsim; the paired exact moment-ODE jobs provide
noise-free references for the same data and parameters.

> Rijal K, Mehta P. "A differentiable Gillespie algorithm for simulating chemical kinetics,
> parameter estimation, and designing synthetic biological circuits." *eLife* 2025;
> 14:RP103877. [doi:10.7554/eLife.103877](https://doi.org/10.7554/eLife.103877)
>
> Jones DL, Brewster RC, Phillips R. "Promoter architecture dictates cell-to-cell variability
> in gene expression." *Science* 2014; 346:1533–1536.
> [doi:10.1126/science.1255301](https://doi.org/10.1126/science.1255301)

## Jobs

**Stochastic (exact SSA) — the primary jobs:**

- [`lacud5_ssa`](lacud5_ssa/): Jones Fig. 3A `lacUV5`, called `lacUD5` by Rijal;
  fits `r_over_gamma` and `gamma`; exact SSA with 200 trajectories/evaluation;
  reproduction PASS and bounded-fit PARTIAL; confidence 82/100.
- [`five_dl1_ssa`](five_dl1_ssa/): Jones Fig. 3A `5DL1`; fits `r_over_gamma` and
  `gamma`; exact SSA with 200 trajectories/evaluation; reproduction and bounded-fit PARTIAL;
  confidence 70/100.

**Deterministic twins (exact moment ODEs) — added as cross-checks, not replacements:**

- [`lacud5_ode`](lacud5_ode/): same data, same two parameters, moments from the exact moment
  equations; `gntr` gradient fit reaches the global optimum in 47 s; confidence 95/100.
- [`five_dl1_ode`](five_dl1_ode/): the same for `5DL1`; 51 s; confidence 95/100.

The SSA jobs are tiny finite networks: three species and four reactions. Each folder is an
independent, self-contained fit so each promoter's published parameter set can be used as the
nominal point.

### Why a deterministic twin is possible here

Every propensity of the two-state model is **linear** in the state and the promoter indicator is
binary (`g² = g`), so the moment hierarchy **closes exactly** on `<g>`, `<m>`, `<gm>`, `<m²>` — no
moment-closure approximation. That linear system has non-negative couplings, so it is itself a
mass-action network, and the `*_ode` model files encode it directly: **their species are the
moments.** Integrating them deterministically yields exactly the `E[m]` and `E[m²]` the SSA jobs
estimate by path sampling, with no Monte Carlo error and a smooth, differentiable objective.

This is a property of *this topology*, not a general substitute for SSA. A bimolecular repression
step would break the closure. The twins exist to give the stochastic jobs a noise-free reference —
and, in passing, the exact global optimum of Rijal Eq. 14 under this reduction:

| promoter | Rijal Fig. 7 sos | twin's optimum sos | at |
|---|---|---|---|
| lacUD5 | 5.177577 | **4.393239** | `r/gamma` 13.6086, `gamma` 8.1503 |
| 5DL1 | 4.216492 | **0.416911** | `r/gamma` 7.4024, `gamma` 5.9906 |

Both were confirmed to be global by a dense closed-form grid scan over the same box. Note the
objective convention: PyBNF's edition-2 `sos` desugars to `Gaussian(sigma = 1)`, whose kernel
carries a factor ½, so **Rijal Eq. 14 = 2 × the objective PyBNF reports**.

A structural consequence of the `kon_R` reduction below, visible once the moments are exact:
`<m> = mean_target` **identically, for any `r_over_gamma` and `gamma`**, so the `Mean_mRNA` residual
is always zero and the fit is driven entirely by the SD/Fano column. In the SSA jobs that same term
contributes only Monte Carlo noise.

## Model and stochastic measurement

The promoter switches between active and repressor-bound states,

```text
active --kon_R--> inactive       active --r--> active + mRNA
inactive --koff_R--> active      mRNA --gamma--> 0
```

with `koff_R = 1`, as in Rijal and Mehta. Each BNGsim SSA trajectory reports the terminal mRNA
count `m` and `m^2`. PyBNF's `smoothing` averages these over independent trajectories, after which
edition-2 measurement formulas construct

```text
Mean_mRNA = E[m]
mRNA_SD   = sqrt(E[m^2] - E[m]^2).
```

The ordinary `sos` objective is therefore Rijal Eq. 14: squared error in the experimental mean
plus squared error in the experimental mRNA standard deviation. This uses PyBNF's existing
objective and exact BNGsim SSA; no differentiability or custom optimizer is required.

## Data and the necessary reduction

Jones Fig. 3A reports mean mRNA and Fano factor. The exact numeric arrays staged in each job come
from the [authors' DGA repository][dga].
The `.exp` standard deviation is the lossless transformation
`sqrt(mean * Fano)`. The authors' notebook removes one duplicate-mean row per promoter with
`np.unique(..., return_index=True)`, leaving 17 fit rows from each 18-row source array.

Rijal and Mehta fit one latent `kon_R` for every experimental row, but those fitted on-rates were
not tabulated or released. An edition-2 parameter scan also cannot introduce a distinct fitted
parameter for every row. These jobs eliminate that latent input using the exact steady-state mean:

```text
kon_R = koff_R * (r_over_gamma / mean_target - 1).
```

Thus the measured mean indexes the scan and the optimizer estimates `r_over_gamma` and `gamma`;
the transcription rate is derived as `r = r_over_gamma * gamma`. This preserves the four-reaction
stochastic model and makes the real-data fit runnable, but it is a reduced exact-SSA adaptation,
not a byte-for-byte reproduction of the paper's transient differentiable-Gillespie fit.

## What is verified

- Both edition-2 configurations parse and bind the two fitted parameters without legacy
  `__FREE` aliases.
- Both export, lint, and import through PEtab v2 structurally. PEtab does not encode PyBNF's
  SSA-replicate smoothing semantics, so this is a schema round-trip, not a claim that a
  deterministic PEtab runner computes the same ensemble moments.
- A bounded DE smoke fit reaches a finite objective through `BngsimModel`, `method=ssa`, and
  200-trajectory smoothing in about nine seconds per job on the validation machine.
- The committed reproduction figures use 2,000 exact SSA trajectories at the published Fig. 7
  parameter values. See each `VALIDATION.md` for metrics and limitations.

The `lacUD5` spelling follows Rijal's paper and released array. Jones calls the same promoter
series `lacUV5`; the local source table retains the Rijal spelling and documents this discrepancy.

## Run

```bash
export BNGPATH=/path/to/BioNetGen

cd examples/real-world/Rijal-2025/lacud5_ssa
pybnf -c lacud5_ssa.conf

cd ../five_dl1_ssa
pybnf -c five_dl1_ssa.conf

# deterministic twins (seconds, gradient-fitted)
cd ../lacud5_ode
pybnf -c lacud5_ode.conf

cd ../five_dl1_ode
pybnf -c five_dl1_ode.conf
```

Each job's `make_data.py` regenerates its `.exp` file from the staged source table, and
`make_reproduction.py` regenerates its exact-SSA comparison figure.

[dga]: https://github.com/Emergent-Behaviors-in-Biology/Differentiable-Gillespie-Algorithm
