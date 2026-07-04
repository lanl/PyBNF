# Lesson 34 — Arithmetic `observableFormula` in a PEtab table

**Feature:** a multi-term arithmetic `observableFormula` (ratio / log / scale) carried in a standard PEtab `observables.tsv`, and its round-trip · **Difficulty:** ★★★ · **Tier:** default CI (import + round-trip) + recovery (fit)

[Lesson 14](../14_observable_layer) wrote measurement-model formulas — a fraction,
a log, a scale — **natively** in a `.conf`. The PEtab lessons so far used only the
*simple* forms of `observableFormula`: a bare observable name
([Lesson 12](../12_petab_roundtrip)) or a per-observable gain/noise parameter
([Lesson 20](../20_petab_observable_parameters)). This lesson closes the gap: a
genuinely **arithmetic** `observableFormula` — a multi-term expression over the
model's observables and parameters — living in a standard PEtab
[`observables.tsv`](observables.tsv), and how it imports and round-trips.

The key idea (ADR-0036): a PEtab `observableFormula` is a **measurement model** — a
post-simulation transform over the output trajectory — **not** an edit to the
model. The [`chain.bngl`](chain.bngl) model declares only the raw species
`Obs_A`/`Obs_B`/`Obs_C`; every measured quantity is a formula the PEtab problem
supplies on top.

## The observables table

`chain.bngl` is a two-step conversion `A → B → C` (rates `k1`, `k2`, plus a fixed
volume `Vd`). Its [`observables.tsv`](observables.tsv) measures three *derived*
quantities, one of each arithmetic flavor:

| observableId | `observableFormula` | kind | what it is |
|--------------|---------------------|------|------------|
| `frac_C` | `Obs_C / (Obs_A + Obs_B + Obs_C)` | **ratio** | the dimensionless fraction that has reached C |
| `log_A`  | `ln(Obs_A)` | **log** | linearizes A's exponential decay → pins `k1` |
| `conc_B` | `Obs_B / Vd` | **scale** | a concentration, amount ÷ volume (a *model parameter*) |

Two things this shows beyond a bare-name observable: a formula can combine several
observables (`frac_C`), and it can reference a model **parameter** (`Vd` in
`conc_B`), not only species.

> **PEtab math spelling.** The formulas use PEtab's math grammar — note `ln` for
> the natural log. (`log10` parses in PEtab, but the measurement layer's
> `lambdify` compiler rejects it; `ln`, or `log(x)/log(10)`, is the portable way to
> take a log — the same rule as [Lesson 14](../14_observable_layer).)

## Import it

```python
from pybnf.petab import import_job
import_job("problem.yaml", "out")
```

Each arithmetic `observableFormula` imports to a native **measurement-model line**
(the [Lesson 14](../14_observable_layer) syntax) — the model file carried verbatim,
nothing synthesized into it:

```
observable: frac_C, formula: Obs_C / (Obs_A + Obs_B + Obs_C)
observable: log_A,  formula: ln(Obs_A)
observable: conc_B, formula: Obs_B / Vd
```

The imported `.exp` columns are named for the observables (`frac_C`, `log_A`,
`conc_B`), reconstructing [`measurements.tsv`](measurements.tsv) exactly.

## It round-trips

Import then re-export, and each `observableFormula` comes back denoting the same
function:

```
observables.tsv  --import-->  observable: … formula: …  --export-->  observables.tsv
```

That fidelity — a measurement model is a first-class, round-trippable part of a
PEtab problem — plus an end-to-end fit that recovers `k1`/`k2` by materializing
each formula over the real bngsim trace, is what
[`tests/test_tutorial_petab_observable_formula.py`](../../../tests/test_tutorial_petab_observable_formula.py)
checks (import + round-trip are backend-free/default CI; the fit is recovery-tier).

## Where this sits

- [Lesson 14](../14_observable_layer) — the same ratio/log/scale measurement models
  written *natively* in a `.conf` (this lesson is their PEtab-table companion; the
  native framing there is deliberate and unchanged).
- [Lesson 12](../12_petab_roundtrip) — a bare-name `observableFormula` (an
  observable measured directly).
- [Lesson 20](../20_petab_observable_parameters) — per-observable `observableParameters`
  (an *estimated* gain) and `noiseParameters`, the other kind of formula content.
- [Lesson 33](../33_sbml_petab) — a bare-species `observableFormula` on an **SBML**
  PEtab problem (the trivial-formula case this lesson generalizes).

## Regenerating the fixtures

```bash
python regenerate_fixtures.py      # needs bngsim + BNG2.pl (set BNGPATH)
```

This simulates `chain.bngl` at the true rates, materializes each formula with the
fit's own measurement-layer code, and rewrites the PEtab tables — so the data is
always the model's own observed output at the truth.
