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
defect. The **How** column says whether the linter *errors* (rejects), only
*warns* (advises), or the defect is *rejected at load* before lint runs — see
[Three kinds of "invalid"](#three-kinds-of-invalid) below.

| Fixture | The defect | Caught by | How |
| --- | --- | --- | --- |
| [`clean`](clean) | *(none — the baseline)* | *lints clean* | — |
| [`undefined_observable`](undefined_observable) | a measurement references an observable the table never defines | `CheckMeasuredObservablesDefined` | error |
| [`observable_shadows_entity`](observable_shadows_entity) | an observable id collides with a model species/observable name | `CheckObservablesDoNotShadowModelEntities` | error |
| [`missing_parameter`](missing_parameter) | an observable formula uses a symbol declared nowhere | `CheckAllParametersPresentInParameterTable` | error |
| [`override_placeholder_mismatch`](override_placeholder_mismatch) | a formula placeholder has no matching measurement override | `CheckOverridesMatchPlaceholders` | error |
| [`bad_condition_target`](bad_condition_target) | a condition perturbs a symbol that is not a model entity | `CheckValidConditionTargets` | error |
| [`bad_prior`](bad_prior) | a `normal` prior is given the wrong number of parameters | `CheckPriorDistribution` | error |
| [`unknown_prior_distribution`](unknown_prior_distribution) | an unrecognized prior distribution name | *rejected at load* | raises |
| [`malformed_bngl`](malformed_bngl) | a BNGL syntax error | `CheckModel` (via `BNG2.pl --check`) | error |
| [`pos_log_measurement`](pos_log_measurement) | a log-normal observable given a non-positive measurement | `CheckPosLogMeasurements` | error |
| [`duplicate_observable_id`](duplicate_observable_id) | the observable table repeats a primary key | `CheckUniquePrimaryKeys` | error |
| [`model_entity_as_parameter`](model_entity_as_parameter) | a model entity placed in the parameter table | `CheckValidParameterInConditionOrParameterTable` | error |
| [`measurement_bad_model_id`](measurement_bad_model_id) | a measurement names a `modelId` no model defines | `CheckMeasurementModelId` | error |
| [`missing_config_file`](missing_config_file) | the problem omits the required `parameter_files` | *rejected at load* | raises |
| [`missing_experiment_condition`](missing_experiment_condition) | an experiment applies a condition the table lacks | `CheckExperimentConditionsExist` | error |
| [`undefined_experiment`](undefined_experiment) | a measurement names an experiment no table defines | `CheckUndefinedExperiments` | **warning** |
| [`unused_experiment`](unused_experiment) | the experiment table defines an experiment nothing uses | `CheckUnusedExperiments` | **warning** |
| [`unused_condition`](unused_condition) | the condition table defines a condition nothing applies | `CheckUnusedConditions` | **warning** |
| [`initial_change_symbol`](initial_change_symbol) | a `t=0` condition sets a target from an outside symbol | `CheckInitialChangeSymbols` | error |

The last ten fixtures were added to cover the rest of petab's default lint task
set (issue #420). Five of them introduce two tables the baseline never needed —
`experiments.tsv` (`experimentId, time, conditionId`) and `conditions.tsv`
(`conditionId, targetId, targetValue`), wired in through the problem YAML's
`experiment_files:` / `condition_files:` keys. A couple of defects necessarily
trip a sibling check too (a model entity in the parameter table is *also* an
extraneous parameter); the test asserts the named task is **among** those that
flagged, so a co-firing sibling is fine.

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

## Three kinds of "invalid"

Not every mistake surfaces the same way, and the distinction is worth learning:

- **Lint errors** (most fixtures) — the problem *loads*, but `lint_problem` returns
  a report with `has_errors() == True`. These are semantic cross-checks between
  tables (a measurement naming a nonexistent observable, a formula naming an
  undeclared parameter, …).
- **Lint warnings** (`undefined_experiment`, `unused_experiment`, `unused_condition`)
  — the report contains items, but `has_errors()` stays `False`: petab *advises*
  rather than rejects. An unused experiment or a measurement pointing at an
  undefined one is suspicious, not fatal, so these come back at `WARNING` level.
  Read a report item's `.level` (a `ValidationIssueSeverity`) to tell the two
  apart — `has_errors()` only counts `ERROR` and above.
- **Load-time rejection** (`unknown_prior_distribution`, `missing_config_file`) —
  petab's tables and problem config are a *typed* (pydantic + JSON-schema) model,
  so a structurally impossible input — a bogus distribution name, or a problem
  missing a required file section — is rejected the instant `Problem.from_yaml`
  parses it, before lint ever runs. That early rejection is the validator doing its
  job too, just at a different layer.

## Two checks this gallery can't fake (petab 0.8.2)

Two tasks in petab's default set can't be provoked through a file-based problem, so
— rather than ship a fixture that pretends to trigger them — the clinic documents
why (they're recorded in `_manifest.LINT_UNCOVERED`):

- **`CheckExperimentTable`** (duplicate timepoints within an experiment) —
  `ExperimentTable.from_df` groups rows by `experimentId` and iterates
  `df[time].unique()`, so two rows at the same time collapse into *one* period. A
  duplicate timepoint is therefore unreachable from a table; it would take a
  hand-built `Experiment` object to construct one.
- **`CheckMeasuredExperimentsDefined`** — not present in petab 0.8.2's
  `default_validation_tasks` at all. Its job (a measurement naming an experiment no
  table defines) is done instead by the *warning*-level `CheckUndefinedExperiments`,
  which the [`undefined_experiment`](undefined_experiment) fixture covers.

That leaves the clinic exercising every petab lint task that a committed
`language: bngl` problem can actually reach.

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
