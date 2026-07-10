# Lesson 11 — Model interop (one model, three languages, one backend)

**Feature:** fitting SBML and Antimony models on the bngsim backend · **Difficulty:** ★★☆

PyBNF is a rule-based tool, but it doesn't only speak BNGL. The **bngsim** backend
also ingests **SBML** (the community exchange format — think a model downloaded from
[BioModels](https://www.ebi.ac.uk/biomodels/)) and **Antimony** (a compact,
human-readable language for SBML). This lesson writes the *same* one-step
conversion `A → B` three ways and fits all three to the same data — recovering the
same rate `k`, down to the same objective value.

## The same model, three files

| File | Language | How PyBNF handles it |
| --- | --- | --- |
| [`decay.bngl`](decay.bngl) | BNGL | native; `bngl_backend = bngsim` |
| [`decay.ant`](decay.ant) | Antimony | converted to SBML at load, simulated by bngsim |
| [`decay.xml`](decay.xml) | SBML | ingested directly, simulated by bngsim |

All three encode `A(t) = A0·exp(−k·t)` with `A0 = 100`, `k = 0.5`.

## The three fits

Each is fit to the same [`decay.exp`](decay.exp):

```bash
pybnf -c fit_bngl.conf
pybnf -c fit_antimony.conf
pybnf -c fit_sbml.conf
```

All recover `k ≈ 0.5` (asserted in
[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py)) — in
fact to an *identical* objective, because they are the same dynamics.

## Two things the SBML/Antimony fits need

1. **`sbml_backend = bngsim`.** In the new era, SBML and Antimony models simulate
   through bngsim (PyBNF selects the path from the `.xml` / `.ant` extension).
2. **A measurement model to name the observable.** An SBML/Antimony model reports
   its **species** (here `A`), not named observables. So the confs add a one-line
   measurement model (Lesson 14) to expose the species as the data's column:

   ```
   observable: Obs_A, formula: A
   ```

   The BNGL model needs no such line — its `Obs_A` observable already matches the
   data column. This is exactly why the measurement layer exists: it lets a
   species-based model score against a named data column without editing the model.

## A note on measurement times

The data here is sampled every **half** time unit (0, 0.5, 1.0, …, 4.0), so most
points fall at **non-integer** times. The SBML/Antimony simulation reports at exactly
the experiment's measurement times — the same as the native BNGL path — so any
spacing works and all three fits recover the same `k`. (Earlier releases simulated
SBML/Antimony on a uniform integer grid and silently dropped off-grid times like
`t = 0.5`; that was fixed in #469/#470.)

## The takeaway

A model you already have in SBML or Antimony is a first-class PyBNF citizen — you
don't have to re-encode it in BNGL to fit it. Point a `.conf` at the file, add a
measurement model to name the observable, and fit it exactly as you would a BNGL
model.
