# Lesson 33 — Importing an SBML PEtab problem

**Feature:** import a standard **SBML** PEtab v2 problem and fit it through the new-era in-process bngsim engine · **Difficulty:** ★★★ · **Tier:** default CI (import) + recovery (fit)

Every PEtab lesson so far ([12](../12_petab_roundtrip), [15](../15_petab_priors),
[20](../20_petab_observable_parameters), [29](../29_petab_protocols)) has used a
model written in **BNGL**. But PEtab's native modeling language is **SBML**, and
the reason PyBNF's bngsim backend was made fully SBML-compliant is exactly this:
to import a standard SBML PEtab problem — the format the systems-biology community
publishes benchmarks in — and fit it directly, with the SBML model carried through
**untouched**.

## The model: a three-state cycle, in SBML

The dynamical model is a three-state irreversible cycle `X1 → X2 → X3 → X1` with a
single shared rate `k`, written compactly in [Antimony](cycle.ant) and converted
to [SBML](cycle.xml):

```
species X1 = 90, X2 = 10, X3 = 0;
k = 1.0;
r12: X1 -> X2;  k*X1;
r23: X2 -> X3;  k*X2;
r31: X3 -> X1;  k*X3;
```

The three species chase each other around the cycle and relax with a **damped
oscillation** to the uniform steady state (`X1 = X2 = X3 = 33.33`): `X1` overshoots
downward, `X2`/`X3` rise and cross, all settling by `t ≈ 5`. The rate `k` sets both
the relaxation rate (`3k/2`) and the oscillation frequency (`√3·k/2`), so a single
time course pins it hard — `k` is the one fitted parameter.

## The PEtab problem

[`problem.yaml`](problem.yaml) ties the standard v2 tables together and declares
the model in **its own language**, pointing at the verbatim `.xml`:

```yaml
model_files:
  cycle:
    location: cycle.xml
    language: sbml
```

| table | what it holds |
|-------|---------------|
| [`parameters.tsv`](parameters.tsv) | `k` is estimated, bounds `[0.1, 5]` |
| [`observables.tsv`](observables.tsv) | two observables, `observableFormula` = `X1` and `X2` (bare species) |
| [`measurements.tsv`](measurements.tsv) | the cycle's own output at `k = 1.0`, on the integer grid `t = 0..8` |
| [`conditions.tsv`](conditions.tsv) / [`experiments.tsv`](experiments.tsv) | header-only — this problem has no perturbations (a single time course under the model's own initial state; [Lessons 9](../09_experiment_design) and [29](../29_petab_protocols) fill these in for dose-response and washout) |

## Import it

```python
from pybnf.petab import import_job
import_job("problem.yaml", "out")
```

`import_job` produces a runnable conf ([`out/imported.conf`](../12_petab_roundtrip))
and one `.exp` per experiment, with two things worth noting:

- **The `.xml` is carried byte-verbatim.** The dynamical model is never edited by
  the importer (ADR-0036) — `out/cycle.xml` is identical to the input.
- **Bare-species observables become direct species measurements.** Because each
  `observableFormula` is a plain species id (`X1`, `X2`) and the bngsim SBML path
  reports *raw species* (Lessons [11](../11_interop)/[31](../31_bngl_sbml_fit)),
  the imported `.exp` columns are simply `X1` and `X2` — no measurement-model line
  is needed. (An *arithmetic* `observableFormula` — a ratio, a scale — would
  instead import as an `observable: … formula: …` line; that is
  [Lesson 34](../34_petab_observable_formula).)

The imported conf:

```
model: cycle.xml
job_type = de
objective = sos
experiment: exp1, method: ode, data: exp1.exp
uniform_var = k 0.1 5
```

## Run it with the new-era engine

By default an SBML fit uses the `roadrunner` backend. The **new-era, in-process**
path — the whole point of this lesson — is bngsim, opted into with one line
(exactly as in [Lesson 11](../11_interop)):

```
sbml_backend = bngsim
```

Run the imported job that way and it recovers the true rate from the
damped-oscillation data:

```
k → 1.0     (objective → 0)
```

That full round trip — a standard SBML PEtab problem in, a recovered parameter out,
the SBML simulated in-process by bngsim — is what
[`tests/test_tutorial_sbml_petab.py`](../../../tests/test_tutorial_sbml_petab.py)
checks: the problem lints clean and imports (backend-free, default CI), and the
imported job recovers `k = 1.0` through bngsim (recovery tier).

> **Integer grid only (for now).** The new-era `experiment:` surface does not yet
> thread a *non-integer* data grid into an SBML simulation
> ([lanl/PyBNF#470](../11_interop)), so this problem is authored on `t = 0..8`.
> Native BNGL models thread any grid fine; the constraint is specific to the
> SBML/Antimony simulation path.

## Where this sits

- [Lesson 11](../11_interop) — the same A→B reaction as BNGL, SBML, and Antimony,
  all fit through bngsim; where `sbml_backend = bngsim` is introduced.
- [Lesson 31](../31_bngl_sbml_fit) — a single fit mixing a BNGL model *and* an SBML
  model together.
- [Lessons 12](../12_petab_roundtrip)/[15](../15_petab_priors)/[20](../20_petab_observable_parameters)/[29](../29_petab_protocols)
  — the same PEtab machinery on BNGL models (round-trip, priors, observable
  parameters, protocols).
- [Lesson 34](../34_petab_observable_formula) — arithmetic `observableFormula` in a
  PEtab observables table (the expression case this lesson's bare-species
  observables sidestep).

## Regenerating the fixtures

```bash
python regenerate_fixtures.py      # needs antimony + bngsim + BNG2.pl (set BNGPATH)
```

This converts `cycle.ant → cycle.xml` and re-simulates the SBML at `k = 1.0`
through bngsim to refill `measurements.tsv`, so the PEtab data is always the
model's own output at the truth.
