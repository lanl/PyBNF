# Lesson 12 — Interoperate with PEtab v2 (and lint BNGL models)

**Feature:** PEtab v2 export / import / validation; the BNGL PEtab linter · **Difficulty:** ★★☆

[PEtab](https://petab.readthedocs.io) is a community standard for specifying
parameter-estimation problems (model + data + observables + parameters) in a
tool-independent way. PyBNF speaks **PEtab v2**, and — uniquely — it can use a
**BNGL** model as the PEtab model, via a small loader it registers into the
`petab` library (`pybnf.petab.bngl_model.register_bngl`). This lesson shows the
full round trip and the validation ("lint") path.

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

## Lint it (dogfood the BNGL loader)

Because PyBNF registers a BNGL loader, the standard `petab` validator can load
and check a `language: bngl` problem:

```python
from pybnf.petab.bngl_model import register_bngl
from petab.v2 import Problem
from petab.v2.lint import lint_problem

register_bngl()                                  # teach petab about BNGL
problem = Problem.from_yaml("petab/problem.yaml")
report = lint_problem(problem)
assert not report.has_errors()                   # cross-checks pass
```

The model-level validity check shells out to `BNG2.pl --check` (the real BNGL
validator) when a BioNetGen is available, and degrades gracefully to "valid"
when it isn't — so validation never falsely fails for lack of a backend.

> This is the linter we intend to contribute upstream to
> [libpetab-python](https://github.com/PEtab-dev/libpetab-python); exercising it
> across the tutorial models (and the analytical-ODE catalog) is how we build
> confidence in it first.

## Import a PEtab v2 problem → a runnable PyBNF job

The reverse recovers a runnable job. PEtab fixes the *problem* (model, data,
parameters) but says nothing about *how to search* it, so you supply the recipe
(`job_type`, algorithm settings):

```python
from pybnf.petab import import_job
import_job("petab/problem.yaml", "imported/", job_type="de")
# -> imported/imported.conf + imported/*.exp + the model, ready for `pybnf -c`
```

## What to notice

- **The same model, three representations** — a `.bngl` you fit directly (Lesson
  2), a PEtab v2 problem you exchange with other tools, and an imported job you
  run again. They round-trip.
- **The BNGL loader is the interesting bit.** PEtab shipped only SBML/PySB model
  loaders; PyBNF adds BNGL, so a rule-based modeller can use the whole PEtab
  ecosystem. The tests in `tests/test_tutorial_petab.py` validate every exported
  problem, which is exactly the experience we want before proposing it upstream.
