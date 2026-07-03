# Lesson 20 — PEtab per-observable parameters (gains and noise you import)

**Feature:** importing PEtab `observableParameters` / `noiseParameters` · **Difficulty:** ★★★

A real measurement is rarely the model observable itself. Each **channel** of an
instrument has its own **gain** (an unknown scale factor) and its own **noise
level**, and neither belongs to the biology — they belong to the measurement.
PEtab expresses these as per-observable **`observableParameters`** (the gain) and
**`noiseParameters`** (the noise), often *estimated*. This lesson shows how PyBNF
**imports** them into its native config surface — the same machinery that reads a
real benchmark like Boehm, distilled to a tiny BNGL problem.

## The problem

One chain, `A --k1--> B --k2--> C`, read on **two channels** (`twochannel.bngl`,
observing `Obs_B` and `Obs_C`). The committed PEtab v2 problem gives each channel a
placeholder gain and a placeholder noise, and the measurement table binds them to
estimated parameters:

```
# observables.tsv (per-channel placeholders)
obs_B   observableParameter1_obs_B * Obs_B   …   noiseParameter1_obs_B   …   normal
obs_C   observableParameter1_obs_C * Obs_C   …   noiseParameter1_obs_C   …   laplace

# measurements.tsv (constant-per-observable overrides)
obs_B   …   observableParameters=scale_B   noiseParameters=sd_B
obs_C   …   observableParameters=scale_C   noiseParameters=sd_C

# parameters.tsv — every gain and noise is estimated, alongside the model rates
k1, k2, scale_B, sd_B, scale_C, sd_C   (all estimate = true)
```

## What the import produces

Run the importer (`pybnf.petab.import_job`) and read the `imported.conf` it writes:

| PEtab construct | Imports to |
| --- | --- |
| `observableParameters = scale_B` in `observableParameter1_obs_B * Obs_B` | `observable: obs_B, formula: Obs_B*scale_B` |
| `noiseParameters = sd_B`, `noiseDistribution = normal` | `noise_model obs_B = gaussian, sigma = fit sd_B` |
| `noiseParameters = sd_C`, `noiseDistribution = laplace` | `noise_model obs_C = laplace, scale = fit sd_C` |
| every estimated `scale_*` / `sd_*` / rate | a `uniform_var = … ` free parameter |

Two things worth noticing:

- **The gain is folded into the observable formula.** A constant-per-observable
  `observableParameters` id becomes a scale multiplying the raw model observable —
  exactly lesson 14's `observable: …, formula: scale * Obs` construct, now arriving
  *from* PEtab instead of hand-written.
- **The noise is per-observable, family and all.** A constant-per-observable
  `noiseParameters` id becomes a native `noise_model <obs> = …, … = fit <id>` line
  (lesson 10's per-observable noise). The `noiseDistribution` picks the family, so a
  Gaussian channel and a Laplace channel import to `gaussian, sigma = fit …` and
  `laplace, scale = fit …` respectively.

The whole-fit `objective = chi_sq` is the structural base; each per-observable
`noise_model` line overrides it channel by channel.

## Why this matters

This is the observation model of real PEtab benchmarks. Boehm 2014, for instance,
has three readouts each with its own estimated `sd_*` — precisely the
`noiseParameters` pattern here — and PyBNF imports it end-to-end (see
`tests/test_petab_import.py::TestRealWorldBoehmV2`). This lesson is that mechanism
in miniature, and its import is asserted line-by-line in
[`tests/test_tutorial_petab_import.py`](../../../tests/test_tutorial_petab_import.py).

## Regenerating the fixture

The PEtab tables are produced from the manifest's channel cases (so the fixture and
the expected import can't drift):

```bash
python regenerate_fixtures.py     # rewrites problem.yaml + the three tables
```

`OBS_PARAM_CASES` in [`_manifest.py`](../_manifest.py) is the single source of truth
for both the tables this writes and the import the test asserts.
