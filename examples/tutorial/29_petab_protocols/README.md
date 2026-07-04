# Lesson 29 — PEtab protocols: conditions & experiments round-trip

**Feature:** exporting/importing experimental *protocols* (dose-response, pre-equilibration) via PEtab v2 `conditions`/`experiments` tables · **Difficulty:** ★★★ · **Tier:** default CI

[Lesson 12](../12_petab_roundtrip) round-tripped a plain time course through PEtab.
But real fits carry *protocols* — a dose-response scan, a two-phase washout, a
pre-equilibration — and [PEtab v2](https://petab.readthedocs.io) has two extra
tables for exactly that: **`conditions.tsv`** (a named set of parameter/state
values) and **`experiments.tsv`** (a sequence of timed *periods*, each applying a
condition). This lesson shows PyBNF's protocols surviving a full round-trip through
those tables — the same two designs as [lesson 9](../09_experiment_design), now
exported to PEtab and imported back.

Everything here is **backend-free** (a BNGL model loads without a simulator), so the
round-trip runs in default CI. There is **no `--petab` CLI flag** — export and import
are the Python functions `pybnf.petab.export_job` / `import_job`.

## Dose-response → conditions + experiments

`dose_response.conf` sweeps the stimulus `k_prod` across five doses, reading the
steady-state level at each. A scan has no home in a plain measurements table, so it
exports to the **dual conditions/experiments shape** (ADR-0046): each dose is a
Condition and an Experiment, and the steady-state readout is a measurement at
`time = inf`.

```python
from pybnf.petab import export_job, import_job
export_job("dose_response.conf", "petab_out")     # writes conditions.tsv, experiments.tsv, ...
```

```
# petab_out/conditions.tsv          # petab_out/experiments.tsv
conditionId          targetId targetValue    experimentId    time  conditionId
cond_doseresponse_0  k_prod   1              doseresponse_0  0     cond_doseresponse_0
cond_doseresponse_1  k_prod   2              doseresponse_1  0     cond_doseresponse_1
...  (one per dose)                          ...  (one per dose)

# petab_out/measurements.tsv  — measured at steady state
observableId  experimentId    time  measurement
obs_A_tot     doseresponse_0  inf   0.5
obs_A_tot     doseresponse_1  inf   1
...
```

Import it back and PyBNF re-infers the parameter scan — the recovered `.exp` is
indexed by the swept parameter `k_prod`, not time:

```python
import_job("petab_out/problem.yaml", "back", job_type="de")
# back/doseresponse.exp  ->  header "# k_prod  A_tot", rows 1..16 at steady state
```

## Washout → a pre-equilibration period

`washout.conf` equilibrates with the stimulus ON, then measures with it OFF. This
round-trips **most faithfully**: each `condition:` becomes a PEtab Condition, and
the experiment becomes two *periods* — the pre-equilibration at PEtab's special
`time = -inf`, then the measurement at `time = 0`:

```
# experiments.tsv
experimentId  time   conditionId
washout       -inf   cond_stim_on      # pre-equilibration period (state carries over)
washout       0      cond_stim_off     # measurement period
```

Import rebuilds the original experiment line exactly:

```
experiment: washout, preequilibrate: stim_on, condition: stim_off, method: ode, data: washout.exp
```

So PEtab's `time = -inf` period ⇄ PyBNF's `preequilibrate:` — a lossless round-trip
of the whole two-phase protocol.

## What round-trips, and what doesn't

The imported job's header says it plainly: the **PEtab problem** — parameters,
observables, measurements, conditions, experiments — is recovered exactly. The
**run recipe** (the `job_type`, algorithm settings, and each experiment's numerical
`method:`) is *supplied* on import, not recovered: PEtab is a problem specification
and has no slot for how you chose to solve it. That is why `import_job` takes a
`job_type` and `settings`.

## Run it

```bash
pytest tests/test_tutorial_petab_protocols.py
```

[`tests/test_tutorial_petab_protocols.py`](../../../tests/test_tutorial_petab_protocols.py)
exports each conf, asserts the conditions/experiments tables are present and the
problem lints clean (dogfooding PyBNF's BNGL PEtab linter), then imports it back and
checks the recovered protocol — the dose-indexed scan and the reconstructed
pre-equilibration. A `BNG2.pl --check` model-validation step runs too when `BNGPATH`
is set (skipped otherwise).

## Where this sits

- [Lesson 12](../12_petab_roundtrip) — the PEtab round-trip for a plain time course,
  and the BNGL linter.
- [Lesson 9](../09_experiment_design) — the same two protocols, used there to *fit*
  a rate constant two ways; here they are the subject of the interchange.
- [Lessons 15](../15_petab_priors) & [20](../20_petab_observable_parameters) — other
  PEtab import surfaces (priors; per-observable scale/noise parameters).
