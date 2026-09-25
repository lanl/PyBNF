# Lesson 12 — Interoperate with PEtab v2 (and lint BNGL models)

**Feature:** PEtab v2 export / import / validation; the BNGL PEtab linter · **Difficulty:** ★★☆

[PEtab](https://petab.readthedocs.io) is a community standard for specifying
parameter-estimation problems (model + data + observables + parameters) in a
tool-independent way. PyBNF speaks **PEtab v2**, and it can use a **BNGL**
model as the PEtab model: the `petab` library loads `language: bngl` natively
(since petab 0.9.0, through a loader PyBNF contributed upstream). This lesson
shows the full round trip and the validation ("lint") path.

## What a PEtab v2 problem looks like

The [`petab/`](petab) folder is a complete PEtab v2 problem, exported from
Lesson 2's Bateman fit:

| File | Contents |
| --- | --- |
| [`petab/problem.yaml`](petab/problem.yaml) | ties the tables together; declares the model with `language: bngl`. |
| [`petab/parameters.tsv`](petab/parameters.tsv) | the free parameters (`k1`, `k2`), bounds, `estimate` flag. |
| [`petab/observables.tsv`](petab/observables.tsv) | one row per observable (`Obs_A/B/C`), its formula and noise model. |
| [`petab/measurements.tsv`](petab/measurements.tsv) | the data, in PEtab's long format. |
| [`petab/bateman_chain.bngl`](petab/bateman_chain.bngl) | the model, carried verbatim. |

## Export a PyBNF job → PEtab v2

Any edition-2 job exports:

```python
from pybnf.petab import export_job
export_job("bateman_chain_de.conf", "petab/")   # run from 02_bateman_chain/
```

## Lint it

Because `petab` loads BNGL natively, its standard validator can load and check
a `language: bngl` problem:

```python
from petab.v2 import Problem
from petab.v2.lint import lint_problem

problem = Problem.from_yaml("petab/problem.yaml")
report = lint_problem(problem)
assert not report.has_errors()                   # cross-checks pass
```

The model-level validity check shells out to `BNG2.pl --check` (the real BNGL
validator) when a BioNetGen is available, and degrades gracefully to "valid"
when it isn't — so validation never falsely fails for lack of a backend.

> This loader was contributed upstream to
> [libpetab-python](https://github.com/PEtab-dev/libpetab-python)
> (PEtab-dev/libpetab-python#508) and ships in petab 0.9.0; exercising it across
> the tutorial models (and the analytical-ODE catalog) is how we built confidence
> in it first.

## Import a PEtab v2 problem → a runnable PyBNF job

The reverse recovers a runnable job. PEtab fixes the *problem* (model, data,
parameters) but says nothing about *how to search* it, so you supply the recipe
(`job_type`, algorithm settings):

```python
from pybnf.petab import import_job
import_job("petab/problem.yaml", "imported/", job_type="de")
# -> imported/imported.conf + imported/*.exp + the model, ready for `pybnf -c`
```

**Fixed parameters.** Every row of this problem's `parameters.tsv` has
`estimate = true`. A row with `estimate = false` fixes that parameter at the row's
`nominalValue`, and PEtab gives the table precedence over the model file. If the row names
a model parameter whose value in the `.bngl` is different, the importer writes the table's
value into its copy of the model, for example
`k1 0.5  # PEtab parameters.tsv: estimate=false, nominalValue 0.5 (model file: 1)`, lists
the change at the top of `imported.conf` and prints it. Your own files are not changed.
The exporter never writes such a row: a parameter a PyBNF job does not fit simply keeps its
value in the exported model, which is what PEtab assumes for a parameter the table leaves
out (ADR-0149).

## What to notice

- **The same model, three representations** — a `.bngl` you fit directly (Lesson
  2), a PEtab v2 problem you exchange with other tools, and an imported job you
  run again. They round-trip.
- **The BNGL loader is the interesting bit.** PEtab shipped only SBML/PySB model
  loaders; PyBNF adds BNGL, so a rule-based modeller can use the whole PEtab
  ecosystem. The tests in `tests/test_tutorial_petab.py` validate every exported
  problem, which is exactly the experience we want before proposing it upstream.
