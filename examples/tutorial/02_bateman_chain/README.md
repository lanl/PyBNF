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
| [`bateman_A_only.exp`](bateman_A_only.exp) | a second dataset with **only** `Obs_A`. |
| [`bateman_chain_profile_likelihood.conf`](bateman_chain_profile_likelihood.conf) | **identifiability** of both rates (all species observed). |
| [`bateman_A_only_profile_likelihood.conf`](bateman_A_only_profile_likelihood.conf) | the same analysis on A-only data — `k2` comes out non-identifiable. |

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

## Bonus — is each rate actually pinned down? (profile likelihood)

A best-fit number isn't the whole story; you also want to know how well the data
*constrains* each parameter. Profile likelihood answers that.

```bash
pybnf -c bateman_chain_profile_likelihood.conf     # all species observed
pybnf -c bateman_A_only_profile_likelihood.conf    # only A observed
```

- **All three species observed** → both `k1` and `k2` are **identifiable**: each
  gets a finite confidence interval that brackets the truth. See
  `Results/…/profile_likelihood_summary.txt`.
- **Only `A` observed** → `A(t) = A0·exp(−k1·t)` doesn't depend on `k2` at all, so
  the data can't constrain it. Profile likelihood reports `k1` **identifiable** but
  `k2` **structurally non-identifiable** (a flat profile; its "interval" spans the
  whole bound). That's the point of the analysis — it tells you which parameters to
  trust before you over-interpret a fit.

## Next

→ **Lesson 3** (`gompertz_growth/`): particle swarm, and the global→local
`refine` recipe on its own.
