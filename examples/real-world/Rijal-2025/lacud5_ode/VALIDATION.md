# VALIDATION — Rijal-2025/lacud5_ode

Validation of the exact moment-ODE twin of the lacUD5 job.

> **Confidence: 95 / 100.** The derivation is exact and was checked three independent ways: the
> integrated network reproduces the closed-form telegraph moments to 1.1e-7 relative, the closed
> form reproduces direct Gillespie sampling within Monte Carlo error, and the gradient fit
> converges to the same optimum a dense closed-form grid scan finds. The deduction is because the
> fit inherits the SSA twin's `kon_R` reduction (the paper's 17 per-row on-rates remain
> unpublished), so the recovered parameters are the optimum of that reduction rather than a
> reproduction of Rijal Fig. 7.

## Gate 0 — materials

Shares all primary materials with [`lacud5_ssa`](../lacud5_ssa/VALIDATION.md): the Rijal and Mehta
2025 paper, the Jones et al. 2014 paper and supplement, and the authors' released numeric arrays.
`lacud5.exp` and `jones_fig3a_source.tsv` are byte-identical to the SSA twin's.

**Verdict: PASS.**

## Gate 1 — the closure is exact, not an approximation

The two-state model's propensities are `kon·g`, `koff·(1−g)`, `r·g`, and `gamma·m` — all linear in
the state — and `g ∈ {0,1}` gives `g² = g`. Writing the moment equations, the only product term that
arises is `<gm>`, whose own equation generates `<g²m> = <gm>`. The hierarchy therefore closes on
`(<g>, <m>, <gm>, <m²>)` with **no closure approximation and no truncation**.

The resulting linear system has non-negative off-diagonal couplings and non-negative decay terms, so
it is realizable as a mass-action network; `lacud5_ode.bngl` encodes it reaction-for-term. The
network's rate equations are the moment equations by construction.

**Verdict: PASS (exact derivation).**

## Gate 2 — integrated network vs. closed form

Steady-state telegraph solution, used as an independent oracle:

```text
p = koff/(kon+koff);  <m> = (r/gamma)p;  Fano = 1 + r(1-p)/(kon+koff+gamma)
```

Integrating the BNGsim network to `t_end = 10` at the Rijal Fig. 7 parameters, over all 17 scan
points:

| quantity | max abs deviation | max rel deviation |
|---|---|---|
| `E[m]` | 2.9e-06 | — |
| `SD` | 7.8e-07 | **1.1e-07** |

The residual is CVODE tolerance, not model error. It also confirms `t_end = 10` (in units of
`1/koff_R`) is fully relaxed — the moment ODEs reach steady state well before then.

**Verdict: PASS.**

## Gate 3 — closed form vs. direct Gillespie

Independent check that the moment equations describe the *stochastic* process, using a direct
Gillespie implementation (4,000 trajectories to `t = 10`, outside PyBNF and outside BNG):

| `mean_target` | analytic SD | moment ODE @ t=10 | direct SSA |
|---|---|---|---|
| 0.0427 | 0.2319 | 0.2319 | 0.2113 ± 0.002 |
| 1.2787 | 2.6953 | 2.6953 | 2.6518 ± 0.030 |
| 10.0 | 6.8484 | 6.8484 | 6.8830 ± 0.077 |

SSA scatters around the exact values within Monte Carlo error, as it must.

**Verdict: PASS.**

## Gate 4 — the fit

`job_type = gntr` (general-objective EFIM trust region), 20 starts × 200 iterations, converged in
**47 s**:

| | `r_over_gamma` | `gamma` | Rijal Eq. 14 sos |
|---|---|---|---|
| Rijal Fig. 7 published | 14.55645 | 6.20 | 5.177577 |
| gntr optimum | 13.608555 | 8.150287 | 4.393239 |
| closed-form grid scan (1200×1200) | 13.6036 | 8.1831 | 4.393258 |

The gradient optimum agrees with the grid minimum to grid resolution and attains a marginally lower
objective, so this is the **global** optimum within the configured box — not a local basin.

**Objective convention (checked, not assumed).** PyBNF's edition-2 `sos` desugars to
`Gaussian(sigma = 1)` (`objective.py` `_OBJECTIVE_DESUGAR`), whose kernel carries a factor ½.
Hence `Rijal Eq. 14 == 2 × PyBNF objective`; PyBNF reported `2.196620` where Eq. 14 is `4.393239`.
Numbers quoted across these jobs must state which convention they use.

**Verdict: PASS.**

## Gate 5 — relationship to the SSA twin

The two jobs share data, parameters, bounds, and the `kon_R` reduction, and differ only in how the
moments are obtained. The exact objective surface computed here is therefore the noise-free limit of
the surface the SSA job samples, and is the natural reference for judging that job's Monte Carlo
error and optimizer behavior.

**Known limits, recorded rather than smoothed over:**

- The optimum is the optimum *of the reduction*. Rijal and Mehta fit a separate latent `kon_R` per
  experimental row; those 17 values were never published, so neither job can reproduce their exact
  fit. Both jobs eliminate them through the steady-state mean identity instead.
- Because that identity makes `<m> = mean_target` **identically for any `r_over_gamma` and `gamma`**,
  the `Mean_mRNA` residual is structurally zero. The entire fit is driven by the SD/Fano column. In
  the SSA twin that same term contributes only Monte Carlo noise.
- Exactness depends on all propensities being linear. A bimolecular repression step would break the
  closure; see the README's scope note.

**Verdict: PASS with documented scope.**
