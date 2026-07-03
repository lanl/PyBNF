# Lesson 1 — Your first fit: logistic growth by gradient descent

**Feature:** gradient-based optimization (`job_type = trf`) · **Backend:** bngsim · **Difficulty:** ★☆☆

The logistic (Verhulst–Pearl) equation describes a population that grows fast
while small and saturates at a carrying capacity `K`:

```
dN/dt = r · N · (1 − N/K)
```

We fit it to a single clean time course and recover the two parameters `r`
(growth rate) and `K` (carrying capacity).

## Files

| File | What it is |
| --- | --- |
| [`logistic_growth.bngl`](logistic_growth.bngl) | The model, in edition-2 form (no `begin actions` block). |
| [`logistic_growth.exp`](logistic_growth.exp) | The data: `Obs_N` sampled at 17 times, noise-free. |
| [`logistic_growth_trf.conf`](logistic_growth_trf.conf) | The fit — every key is commented. |

## Run it

```bash
cd tests/analytical_odes/examples/logistic_growth
pybnf -c logistic_growth_trf.conf
```

The fit drives the sum-of-squares objective to ~0 and lands on the true values
`r = 1.2`, `K = 100` (the values the data was generated at). The best fit is
written to `output/logistic_growth_trf/Results/`.

## Bonus — fitting qualitative data, and model checking

You don't always have a measured curve. Sometimes you only know *qualitative*
facts: "it starts small, reaches capacity, never overshoots." PyBNF fits those
directly, using its Biological Property Specification Language (BPSL).

| File | What it is |
| --- | --- |
| [`logistic_growth.prop`](logistic_growth.prop) | four qualitative properties, in BPSL. |
| [`logistic_growth_constraints.conf`](logistic_growth_constraints.conf) | **fit** to the properties alone (no numbers). |
| [`logistic_growth_check.conf`](logistic_growth_check.conf) | **check** whether the model satisfies them (`job_type = check`). |

```bash
pybnf -c logistic_growth_constraints.conf   # finds r, K that satisfy every property
pybnf -c logistic_growth_check.conf          # reports "Satisfied 4 out of 4 constraints"
```

- A **constraint-only experiment** uses `data: …prop` (not `.exp`) and a `t_end:`
  (there's no measurement grid to borrow, so you say how long to integrate).
- Each property becomes a penalty; a parameter set that satisfies all of them
  scores **0**.
- **`job_type = check`** doesn't fit — it evaluates the model as written and
  counts satisfied properties. The complement of fitting.

## What to notice

- **`edition = 2`** turns on the modern config surface — `model:`,
  `job_type`, `objective`, `experiment:`, and bind-by-name free parameters.
- **The model has no simulation actions.** PyBNF builds the simulation from the
  experiment's `time` column, so the model file only describes the biology.
- **`job_type = trf`** is a *gradient* optimizer. It needs `bngl_backend = bngsim`
  (the source of the parameter sensitivities it consumes) and a *smooth* model.
  Lesson 6 shows what happens when the model is not smooth.

## Next

→ **Lesson 2** (`bateman_chain/`): differential evolution on a multi-species
chain with several observables.
