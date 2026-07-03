# Lesson 13 — A PEtab lint clinic (see the validator catch mistakes)

**Feature:** PEtab v2 validation (`petab.v2.lint`) through PyBNF's BNGL loader · **Difficulty:** ★★☆

Lesson 12 showed the *happy path*: export a job, lint it, get silence. This lesson
is the opposite — a **gallery of broken problems**, each with exactly one defect,
so you can watch the linter catch each one and learn to read what it says.

Why a whole lesson on *broken* problems? Because PyBNF registers a **BNGL model
loader** into the `petab` library (`pybnf.petab.bngl_model.register_bngl`), the
standard `petab.v2` validator can load and check a `language: bngl` problem — and
we want hard evidence that petab's lint tasks actually catch the mistakes a
BNGL-native problem can make, *before* we propose the loader upstream to
[libpetab-python](https://github.com/PEtab-dev/libpetab-python) (issue #420). This
clinic is that evidence: one fixture per lint task, each asserted in
[`tests/test_tutorial_lint_clinic.py`](../../../tests/test_tutorial_lint_clinic.py).

## The gallery

Every subdirectory is a complete, self-contained PEtab v2 problem built around one
three-line model — an exponential decay `A --k--> 0` (`clean/decay.bngl`). The
[`clean/`](clean) baseline lints without complaint; each sibling injects a single
defect:

| Fixture | The defect | Caught by |
| --- | --- | --- |
| [`clean`](clean) | *(none — the baseline)* | *lints clean* |
| [`undefined_observable`](undefined_observable) | a measurement references an observable the table never defines | `CheckMeasuredObservablesDefined` |
| [`observable_shadows_entity`](observable_shadows_entity) | an observable id collides with a model species/observable name | `CheckObservablesDoNotShadowModelEntities` |
| [`missing_parameter`](missing_parameter) | an observable formula uses a symbol declared nowhere | `CheckAllParametersPresentInParameterTable` |
| [`override_placeholder_mismatch`](override_placeholder_mismatch) | a formula placeholder has no matching measurement override | `CheckOverridesMatchPlaceholders` |
| [`bad_condition_target`](bad_condition_target) | a condition perturbs a symbol that is not a model entity | `CheckValidConditionTargets` |
| [`bad_prior`](bad_prior) | a `normal` prior is given the wrong number of parameters | `CheckPriorDistribution` |
| [`unknown_prior_distribution`](unknown_prior_distribution) | an unrecognized prior distribution name | *rejected at load* |
| [`malformed_bngl`](malformed_bngl) | a BNGL syntax error | `CheckModel` (via `BNG2.pl --check`) |

## Run the linter yourself

```python
from petab.v2 import Problem
from petab.v2.lint import lint_problem
from pybnf.petab.bngl_model import register_bngl

register_bngl()                                    # teach petab about BNGL
report = lint_problem(Problem.from_yaml("undefined_observable/problem.yaml"))
print(report.has_errors())                         # -> True
for item in report:
    print(item.task, "|", item.message)            # which Check flagged, and why
```

Each report item carries a **`.task`** (the `petab.v2.lint` `Check` class that
raised it) and a **`.message`** — so you can pin a failure to a specific check,
which is exactly what the test asserts.

## Two kinds of "invalid"

Not every mistake surfaces the same way, and the distinction is worth learning:

- **Lint errors** (most fixtures) — the problem *loads*, but `lint_problem` returns
  a report with `has_errors() == True`. These are semantic cross-checks between
  tables (a measurement naming a nonexistent observable, a formula naming an
  undeclared parameter, …).
- **Load-time rejection** (`unknown_prior_distribution`) — petab's tables are a
  *typed* (pydantic) schema, so a structurally impossible value like a bogus
  distribution name is rejected the instant `Problem.from_yaml` parses it, before
  lint ever runs. That early rejection is the validator doing its job too, just at
  a different layer.

## About `CheckModel`

The `malformed_bngl` fixture is special: model validity is checked by shelling out
to the **real BNGL validator**, `BNG2.pl --check`. Where no BioNetGen is available
(`BNGPATH` unset), PyBNF's loader **degrades gracefully to "valid"** rather than
fail for lack of a backend — so that one fixture's test is skipped without a
BioNetGen, while every table-level check above still runs.

## Regenerating the fixtures

The broken problems are produced by a small dev tool that owns the recipe for each
defect (the committed fixtures are its output):

```bash
python regenerate_fixtures.py     # rewrites clean/ + every broken subdir
```

The *expected* linter reaction for each fixture lives test-side in
[`_manifest.py`](../_manifest.py) (`LINT_CASES`), never in the fixtures — the
generator and the manifest are cross-checked so they can't silently drift.
