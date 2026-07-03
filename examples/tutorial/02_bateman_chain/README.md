# Lesson 2 — Many observables at once: the Bateman chain by differential evolution

**Feature:** metaheuristic optimization (`job_type = de`), multi-observable fitting · **Backend:** bngsim · **Difficulty:** ★☆☆

A sequential decay `A --k1--> B --k2--> C`. We observe all three species and
recover both rate constants at once.

```
A(t) = A0 · exp(-k1 t)
B(t) = A0 · k1/(k2−k1) · ( exp(-k1 t) − exp(-k2 t) )
C(t) = A0 − A(t) − B(t)
```

## Files

| File | What it is |
| --- | --- |
| [`bateman_chain.bngl`](bateman_chain.bngl) | the model (three species, two reactions). |
| [`bateman_chain.exp`](bateman_chain.exp) | data: `Obs_A`, `Obs_B`, `Obs_C` at 21 times. |
| [`bateman_chain_de.conf`](bateman_chain_de.conf) | fit `k1`, `k2` with differential evolution. |
| `bateman_A_only.exp` | a second dataset with **only** `Obs_A` — used by the (upcoming) profile-likelihood lesson to show that `k2` is non-identifiable when B and C aren't observed. |

## Run it

```bash
pybnf -c bateman_chain_de.conf
```

## What to notice

- **`job_type = de`** is differential evolution — a population-based *global*
  optimizer. Unlike Lesson 1's `trf`, it never asks for a gradient, so it makes
  no smoothness demands on the model.
- **One `experiment:` binds three observable columns.** They're scored jointly;
  the fit has to reproduce all three species simultaneously.
- **`refine = 1`** finishes with a local Simplex polish so the noise-free fit
  lands exactly on `k1 = 0.8`, `k2 = 0.25`.

## Next

→ **Lesson 3** (`gompertz_growth/`): particle swarm, and the global→local
`refine` recipe on its own.
