# Lesson 3 — Global search then local polish: Gompertz growth by particle swarm

**Feature:** particle swarm (`job_type = pso`) + refinement (`refine`) · **Backend:** bngsim · **Difficulty:** ★★☆

Gompertz growth, `dX/dt = r·X·ln(K/X)`, with closed form
`X(t) = K·exp(ln(X0/K)·exp(−r t))`. The rate law is nonlinear in `X`, so the
landscape is a little less forgiving than logistic — a good place to see the
standard two-stage recipe PyBNF fits usually finish with: **explore globally,
then polish locally.**

## Files

| File | What it is |
| --- | --- |
| [`gompertz_growth.bngl`](gompertz_growth.bngl) | the model. |
| [`gompertz_growth.exp`](gompertz_growth.exp) | data: `Obs_X` at 21 times. |
| [`gompertz_growth_pso.conf`](gompertz_growth_pso.conf) | fit `r`, `K` with particle swarm + a Simplex refine. |

## Run it

```bash
pybnf -c gompertz_growth_pso.conf
```

## What to notice

- **`job_type = pso`** is Particle Swarm Optimization: candidates ("particles")
  fly through parameter space, pulled toward the best points seen. Asynchronous
  and parallel.
- **`refine = 1` + `refine_method = sim`** runs a local Nelder-Mead Simplex from
  the swarm's best point once the global search finishes. `refine_method` can
  also be `powell` or `cmaes`. This global→local hand-off is how most PyBNF fits
  are finished; it tightens convergence on shallow valleys.

## Next

→ **Lesson 6** (`step_input/`): a model with a discontinuous input — what the
gradient path takes in its stride, what it genuinely refuses, and how to tell
which is which.
