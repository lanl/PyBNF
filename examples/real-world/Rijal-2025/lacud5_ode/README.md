# lacud5_ode — exact moment-ODE twin of `lacud5_ssa`

Deterministic twin of [`lacud5_ssa`](../lacud5_ssa/). It fits the **same data** and the **same two
parameters**, but obtains `E[m]` and `E[m²]` by integrating the process's *exact moment equations*
rather than by averaging exact-SSA trajectories.

This is an **addition, not a replacement**. The SSA job remains the general instrument; this twin
exists because for *this particular* model the moments are available exactly, which makes it a
noise-free reference the stochastic job can be checked against.

## Why the moments are exact here

The two-state promoter model has four reactions:

```text
active --kon_R--> inactive       active --r--> active + mRNA
inactive --koff_R--> active      mRNA --gamma--> 0
```

Every propensity is **linear** in the state, and the promoter indicator `g` is binary, so `g² = g`.
Under those two conditions the moment hierarchy **closes exactly** — the equation for each moment
introduces no higher moment that is not already in the set:

```text
d<g>/dt   = -kon<g> + koff(1-<g>)
d<m>/dt   =  r<g> - gamma<m>
d<gm>/dt  = -kon<gm> + koff(<m>-<gm>) + r<g> - gamma<gm>
d<m^2>/dt = 2r<gm> + r<g> - 2gamma<m^2> + gamma<m>
```

No moment-closure approximation is used and nothing is truncated. This linear system has
non-negative couplings, so it is itself a mass-action reaction network — which is exactly what
`lacud5_ode.bngl` encodes. **Its four species *are* the four moments.** Integrating that network
deterministically therefore yields the ensemble quantities the SSA job estimates by sampling.

At steady state the system has a closed form (the standard telegraph result), which the
reproduction script uses as an independent check:

```text
p    = koff/(kon+koff)
<m>  = (r/gamma) p
Fano = 1 + r(1-p)/(kon+koff+gamma)
```

> **The species are moments, not molecules.** This network is meaningful *only* under ODE
> integration. Simulating it with SSA would be meaningless.

## What this buys

| | `lacud5_ssa` | `lacud5_ode` (this job) |
|---|---|---|
| moments from | 200 SSA trajectories per evaluation | one ODE integration |
| Monte Carlo error | yes (needs `smoothing`) | none |
| objective surface | noisy | smooth, differentiable |
| optimizer | `de` (metaheuristic) | `gntr` (gradient trust region) |
| fit wall time | minutes, thousands of trajectories | **47 s** |

## Result

`gntr` converges to the **global** optimum of Rijal Eq. 14 under this reduction (confirmed against a
1200×1200 closed-form grid scan over the same box):

| | `r_over_gamma` | `gamma` | Rijal Eq. 14 sos |
|---|---|---|---|
| Rijal Fig. 7 published | 14.55645 | 6.20 | 5.177577 |
| this job's optimum | 13.60856 | 8.15029 | **4.393239** |
| closed-form grid minimum | 13.6036 | 8.1831 | 4.393258 |

The fitted point improves on the published one because this job, like its SSA twin, **eliminates
the paper's 17 unpublished per-row `kon_R` values** through the exact steady-state mean identity
(see the parent [README](../README.md)). It is the optimum *of this reduction*, not a correction to
Rijal and Mehta.

**Objective convention.** PyBNF's edition-2 `sos` desugars to `Gaussian(sigma = 1)`, whose kernel
carries a factor ½. So `Rijal Eq. 14 == 2 × (the objective PyBNF reports)`. Both are printed by
`make_reproduction.py` to keep the two scales unambiguous.

## Contents

- `lacud5_ode.bngl` — the moment network (species = `<g>`, `<m>`, `<gm>`, `<m²>`)
- `lacud5_ode.conf` — the edition-2 job (`method: ode`, `job_type = gntr`, no `smoothing`)
- `lacud5.exp` — fit target, byte-identical to the SSA twin's
- `jones_fig3a_source.tsv`, `make_data.py` — data provenance, identical to the SSA twin
- `make_reproduction.py`, `lacud5_ode_reproduction.png` — integrates at the published and fitted
  points and checks both against the closed form
- `VALIDATION.md`

## Running

```bash
BNGPATH=... pybnf -c lacud5_ode.conf
BNGPATH=... python make_reproduction.py
```

## Scope — when this construction stops working

The exactness is a property of the *topology*, not of stochasticity in general. Making repression
bimolecular (an explicit repressor species, `R + P → RP`) would make `d<gm>/dt` depend on `<R g m>`,
the hierarchy would no longer close, and moment closure or SSA would be required. The pseudo-first-
order `kon_R` is what buys exactness here. Likewise, this twin gives the first two moments only — for
full distributions the telegraph model has an analytic steady-state (Beta-Poisson) form, which is a
different object.
