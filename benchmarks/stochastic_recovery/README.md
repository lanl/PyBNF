# Stochastic parameter recovery benchmark

A benchmark for fitting stochastic rule-based models (lanl/PyBNF#663, the small version).
Six published models, each simulated at chosen "true" parameter values to make synthetic
data, and a fit is scored on whether it gets those values back. The right answer is known
exactly, because we chose it.

Design decisions and their reasons are in ADR-0140 (`docs/adr/`).

## The problems

| id | model | reference | free | method | observables | budget (simulations) |
|---|---|---|---:|---|---:|---:|
| `Hlavacek_PNAS2001` | bivalent ligand, receptor dimers, five proofreading steps | Hlavacek et al. 2001, PNAS 98:7295 | 4 | ssa | 3 | 20,000 |
| `Lin_PhysRevE2016` | bursty gene expression with Hill positive feedback | Lin & Doering 2016, Phys Rev E 93:022409 | 4 | ssa | 2 | 20,000 |
| `McKane_PhysRevLett2005` | individual-level predator-prey with demographic noise | McKane & Newman 2005, Phys Rev Lett 94:218102 | 4 | ssa | 2 | 20,000 |
| `Munsky_Science2012` | three two-state promoters with different switching rates | Munsky et al. 2012, Science 336:183 | 6 | ssa | 6 | 20,000 |
| `Shahrezaei_PNAS2008` | promoter switching, transcription, translation, turnover | Shahrezaei & Swain 2008, PNAS 105:17256 | 4 | ssa | 3 | 20,000 |
| `Yang_PhysRevE2008` | trivalent ligand crosslinking bivalent receptors | Yang et al. 2008, Phys Rev E 78:031910 | 3 | nf | 3 | 4,000 |

Problem ids follow the PEtab benchmark collection's `Author_Journal_Year` convention and are
permanent. Every model comes from the curated `BNGL-Models/models` collection (revision
`00e65fa`), where each carries its citation and an independently verified simulation
protocol. The adaptations made here (which parameters are free, which observables are kept,
the sampling window) are written in each model's header and in its `problem.json`.

Each problem directory holds:

* `problem.json`: the frozen definition (format version 1): true values, bounds and
  identifiability of the free parameters; the simulate method and sampling grid; the
  observables; the data's replicate count, seed offset and sigma floor; the simulation budget;
  the replicates per evaluation the baseline methods use.
* `model.bngl`: the model, with the free parameters bound through PyBNF's `name name__FREE`
  alias form and a single simulate action.
* `<suffix>.exp`: the committed data: at each sampling time, the mean of every observable over
  the replicates and a `_SD` column holding the standard deviation across them (floored at
  five percent of the observable's peak mean).

## The protocol

Written in full in `protocol.py`'s docstring. In short:

* **Error** is `|log10(estimate / true)|` per parameter, in decades; a fit's error is the
  largest over the parameters the definition marks identifiable.
* **Success** at the loose tolerance means every identifiable parameter is within a factor of
  two of the truth (0.301 decades); the tight tolerance is 26 percent (0.1 decades). The loose
  tolerance is the headline.
* **Cost** is simulations, and every simulation the fit runs counts: the search, PyBNF's
  end-of-fit confirmation of the best fit (ten candidates run ten more times each, which
  decides the answer), and the replicates run for the information criteria.
  Simulations-to-success is the count when the fit's reported best first came within the loose
  tolerance, reported only for fits whose final answer is within it.
* **Repetition**: every (problem, method) pair runs from several fit seeds; the success rate
  over seeds is the primary statistic, next to the median error and the median
  simulations-to-success.

Every method gets the same simulation budget per problem. The runner enforces it: the fit
stops within one evaluation of the budget, whatever the method's own stopping rule would say.

The data are drawn at replicate indices from one million up. Under PyBNF's default seed policy
a trajectory's seed comes from the parameter values and the replicate index, so a fit that
evaluates the true parameters draws replicates 0, 1, 2, ... of the same process; the offset
keeps it from reproducing the data's own trajectories.

## What the data determine

Every free parameter is marked identifiable, and this table is why. It is the leverage of
each parameter on the frozen objective: how far the objective moves when that parameter
alone is halved or doubled, in units of the objective's own standard deviation at the true
values (six evaluations at the baseline replicate count). From `run_baseline.py leverage`.

| problem | parameter | halved | doubled |
|---|---|---:|---:|
| Shahrezaei_PNAS2008 | `k0` | 2 | 0 |
| Shahrezaei_PNAS2008 | `k1` | 2 | 1 |
| Shahrezaei_PNAS2008 | `v0` | 1 | 16 |
| Shahrezaei_PNAS2008 | `v1` | 1 | 10 |
| Lin_PhysRevE2016 | `B` | 8 | 97 |
| Lin_PhysRevE2016 | `r0` | 1 | 31 |
| Lin_PhysRevE2016 | `r1` | 0 | 17 |
| Lin_PhysRevE2016 | `K` | 13 | 4 |
| McKane_PhysRevLett2005 | `b` | 501 | 860 |
| McKane_PhysRevLett2005 | `d1` | 471 | 1586 |
| McKane_PhysRevLett2005 | `p1` | 1429 | 649 |
| McKane_PhysRevLett2005 | `p2` | 10 | 31 |
| Hlavacek_PNAS2001 | `kon1` | 937 | 1466 |
| Hlavacek_PNAS2001 | `kon2` | 980 | 1425 |
| Hlavacek_PNAS2001 | `koff` | 10806 | 2795 |
| Hlavacek_PNAS2001 | `kp` | 270 | 1015 |
| Munsky_Science2012 | `k_on_I` | 5 | 7 |
| Munsky_Science2012 | `k_off_I` | 7 | 1 |
| Munsky_Science2012 | `k_on_II` | 4 | 6 |
| Munsky_Science2012 | `k_off_II` | 3 | 1 |
| Munsky_Science2012 | `k_on_III` | 5 | 12 |
| Munsky_Science2012 | `k_off_III` | 13 | 6 |
| Yang_PhysRevE2008 | `koff` | 1017 | 1142 |
| Yang_PhysRevE2008 | `kon1` | 919 | 815 |
| Yang_PhysRevE2008 | `kon2` | 20 | 49 |

Every parameter has leverage, but not equally. In the predator-prey problem
(McKane_PhysRevLett2005) `p2`, the predation that does not reproduce, moves the objective
ten to thirty standard deviations where its siblings move it hundreds to a thousand; in the
aggregation problem (Yang_PhysRevE2008) the crosslinking rate `kon2` is twenty to fifty
against about a thousand. Those are narrow directions in an otherwise steep landscape, and
the baseline shows they are what the methods miss. In the gene-expression problems
(Shahrezaei_PNAS2008, Lin_PhysRevE2016, Munsky_Science2012) leverage is low in one direction
for several parameters: halving a promoter or transcription rate that is already slow
changes a replicate mean by less than the noise in ten replicates. A method recovers those
only by averaging over many evaluations, which is what the noise-handling steps are for.

## Baseline methods

| name | fit type | what differs |
|---|---|---|
| `de` | differential evolution | population 20; convergence stop off |
| `ss` | scatter search | reference set 10; `ss_noise_handling = 0` |
| `ss_noise` | scatter search | reference set 10; `ss_noise_handling = 1` (#660) |
| `cmaes` | CMA-ES | population 12; `cmaes_noise_handling = 0` |
| `cmaes_noise` | CMA-ES | population 12; `cmaes_noise_handling = 1` (#661) |

Each pair differs only in the one toggle, so the difference between its two rows is that
feature's measured effect. All fits use the problem's `smoothing` (ten replicates per
evaluation for the SSA problems, four for the NFsim one), the `chi_sq` objective against the
committed `_SD` columns, log-uniform search over the frozen bounds, and no simplex refine.

## Running it

Needs bngsim, BNG2.pl (set `BNGPATH`) and, for `Yang_PhysRevE2008`, bngsim's NFsim backend.

```bash
python benchmarks/stochastic_recovery/run_baseline.py run --seeds 5 --parallel 8 \
    --out benchmarks/stochastic_recovery/results/my_run.json
```

The results file is appended one record per fit and is resumable: a rerun skips the
(problem, method, seed) triples it already holds. Restrict with `--problems`, `--methods`,
`--seeds`, `--budget-scale`. Summarize with

```bash
python benchmarks/stochastic_recovery/run_baseline.py summarize benchmarks/stochastic_recovery/results/my_run.json
```

check that the committed data can still be regenerated from the definitions with

```bash
python benchmarks/stochastic_recovery/run_baseline.py generate --check
```

and measure each parameter's leverage on the objective (the table above) with

```bash
python benchmarks/stochastic_recovery/run_baseline.py leverage
```

A fit of an SSA problem takes two to four minutes on one core at the frozen budget, the NFsim
problem fifteen to twenty-five. The full baseline (six problems, five methods, five seeds)
takes under two hours on eight cores.

From Python:

```python
import sys; sys.path.insert(0, 'benchmarks')
from stochastic_recovery import protocol, harness
problem = protocol.load_problems(ids=['Shahrezaei_PNAS2008'])[0]
record = harness.run_fit(problem, 'cmaes_noise', seed=1)
print(record['max_error'], record['success_loose'], record['simulations'])
```

## Baseline results (v1)

`results/baseline_v1.json` holds the 150 fit records (six problems, five methods, five fit
seeds each, seeds 1 to 5) and `results/baseline_v1.md` this table. Run on 2026-09-11 with
PyBNF at the commit that added the benchmark, bngsim 0.15.1 and BioNetGen 2.9.3. The run is
reproducible: a second run from the same seeds reproduced every estimate, every trace and
every simulation count, and the reported objective values agreed to floating-point rounding
(the last digit of a few of them moves with Python's hash seed, through summation order).

| problem | method | seeds | success (factor 2) | success (26%) | median max error (decades) | median sims to success | mean sims | mean wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Hlavacek_PNAS2001 | cmaes | 5 | 80% | 60% | 0.027 | 2472 | 21014 | 215 |
| Hlavacek_PNAS2001 | cmaes_noise | 5 | 80% | 60% | 0.069 | 3016 | 21015 | 216 |
| Hlavacek_PNAS2001 | de | 5 | 40% | 0% | 0.560 | 5548 | 21019 | 225 |
| Hlavacek_PNAS2001 | ss | 5 | 20% | 0% | 0.843 | 1900 | 21012 | 337 |
| Hlavacek_PNAS2001 | ss_noise | 5 | 20% | 0% | 0.747 | 2000 | 21012 | 305 |
| Lin_PhysRevE2016 | cmaes | 5 | 0% | 0% | 0.977 | - | 21014 | 32 |
| Lin_PhysRevE2016 | cmaes_noise | 5 | 0% | 0% | 0.578 | - | 21015 | 31 |
| Lin_PhysRevE2016 | de | 5 | 0% | 0% | 0.642 | - | 21019 | 39 |
| Lin_PhysRevE2016 | ss | 5 | 0% | 0% | 0.571 | - | 21011 | 79 |
| Lin_PhysRevE2016 | ss_noise | 5 | 0% | 0% | 1.104 | - | 21012 | 92 |
| McKane_PhysRevLett2005 | cmaes | 5 | 0% | 0% | 0.905 | - | 21017 | 192 |
| McKane_PhysRevLett2005 | cmaes_noise | 5 | 40% | 0% | 0.346 | 4006 | 21017 | 194 |
| McKane_PhysRevLett2005 | de | 5 | 0% | 0% | 0.662 | - | 21019 | 225 |
| McKane_PhysRevLett2005 | ss | 5 | 20% | 0% | 0.649 | 21012 | 21012 | 213 |
| McKane_PhysRevLett2005 | ss_noise | 5 | 0% | 0% | 0.951 | - | 21012 | 207 |
| Munsky_Science2012 | cmaes | 5 | 0% | 0% | 0.945 | - | 21016 | 81 |
| Munsky_Science2012 | cmaes_noise | 5 | 0% | 0% | 1.007 | - | 21016 | 78 |
| Munsky_Science2012 | de | 5 | 0% | 0% | 0.936 | - | 21018 | 85 |
| Munsky_Science2012 | ss | 5 | 0% | 0% | 0.840 | - | 21012 | 84 |
| Munsky_Science2012 | ss_noise | 5 | 0% | 0% | 0.890 | - | 21012 | 81 |
| Shahrezaei_PNAS2008 | cmaes | 5 | 20% | 0% | 0.462 | 1241 | 21015 | 110 |
| Shahrezaei_PNAS2008 | cmaes_noise | 5 | 60% | 20% | 0.249 | 470 | 21015 | 107 |
| Shahrezaei_PNAS2008 | de | 5 | 80% | 0% | 0.226 | 1770 | 21019 | 123 |
| Shahrezaei_PNAS2008 | ss | 5 | 40% | 20% | 0.373 | 430 | 21012 | 116 |
| Shahrezaei_PNAS2008 | ss_noise | 5 | 60% | 0% | 0.256 | 4060 | 21012 | 113 |
| Yang_PhysRevE2008 | cmaes | 5 | 20% | 0% | 0.527 | 317 | 4411 | 1017 |
| Yang_PhysRevE2008 | cmaes_noise | 5 | 20% | 0% | 0.699 | 1149 | 4411 | 1080 |
| Yang_PhysRevE2008 | de | 5 | 0% | 0% | 1.151 | - | 4412 | 860 |
| Yang_PhysRevE2008 | ss | 5 | 40% | 0% | 0.560 | 3082 | 4412 | 1375 |
| Yang_PhysRevE2008 | ss_noise | 5 | 0% | 0% | 0.817 | - | 4412 | 1568 |

Per parameter, the median error in decades by method, and how often any method got the
parameter within a factor of two:

| problem | parameter | de | ss | ss_noise | cmaes | cmaes_noise | within a factor of two (all methods) |
|---|---|---:|---:|---:|---:|---:|---:|
| Hlavacek_PNAS2001 | `kon1` | 0.10 | 0.06 | 0.07 | 0.01 | 0.00 | 96% |
| Hlavacek_PNAS2001 | `kon2` | 0.22 | 0.84 | 0.75 | 0.03 | 0.07 | 52% |
| Hlavacek_PNAS2001 | `koff` | 0.07 | 0.39 | 0.35 | 0.02 | 0.04 | 64% |
| Hlavacek_PNAS2001 | `kp` | 0.12 | 0.38 | 0.34 | 0.02 | 0.04 | 60% |
| Lin_PhysRevE2016 | `B` | 0.04 | 0.06 | 0.24 | 0.19 | 0.08 | 96% |
| Lin_PhysRevE2016 | `r0` | 0.40 | 0.18 | 0.50 | 0.24 | 0.23 | 44% |
| Lin_PhysRevE2016 | `r1` | 0.24 | 0.34 | 0.54 | 0.50 | 0.38 | 24% |
| Lin_PhysRevE2016 | `K` | 0.62 | 0.57 | 1.10 | 0.73 | 0.45 | 12% |
| McKane_PhysRevLett2005 | `b` | 0.03 | 0.04 | 0.04 | 0.03 | 0.02 | 88% |
| McKane_PhysRevLett2005 | `d1` | 0.06 | 0.04 | 0.04 | 0.04 | 0.02 | 96% |
| McKane_PhysRevLett2005 | `p1` | 0.06 | 0.04 | 0.04 | 0.04 | 0.02 | 96% |
| McKane_PhysRevLett2005 | `p2` | 0.66 | 0.65 | 0.95 | 0.91 | 0.35 | 12% |
| Munsky_Science2012 | `k_on_I` | 0.44 | 0.40 | 0.18 | 0.22 | 0.35 | 36% |
| Munsky_Science2012 | `k_off_I` | 0.48 | 0.69 | 0.38 | 0.28 | 0.42 | 24% |
| Munsky_Science2012 | `k_on_II` | 0.52 | 0.66 | 0.30 | 0.82 | 0.24 | 28% |
| Munsky_Science2012 | `k_off_II` | 0.61 | 0.72 | 0.35 | 0.94 | 0.41 | 20% |
| Munsky_Science2012 | `k_on_III` | 0.87 | 0.57 | 0.55 | 0.23 | 0.40 | 32% |
| Munsky_Science2012 | `k_off_III` | 0.88 | 0.62 | 0.58 | 0.23 | 0.44 | 32% |
| Shahrezaei_PNAS2008 | `k0` | 0.10 | 0.15 | 0.08 | 0.18 | 0.12 | 92% |
| Shahrezaei_PNAS2008 | `k1` | 0.23 | 0.37 | 0.19 | 0.46 | 0.25 | 52% |
| Shahrezaei_PNAS2008 | `v0` | 0.06 | 0.08 | 0.14 | 0.14 | 0.05 | 100% |
| Shahrezaei_PNAS2008 | `v1` | 0.01 | 0.01 | 0.07 | 0.08 | 0.05 | 100% |
| Yang_PhysRevE2008 | `koff` | 0.07 | 0.06 | 0.04 | 0.04 | 0.05 | 100% |
| Yang_PhysRevE2008 | `kon1` | 0.06 | 0.09 | 0.03 | 0.01 | 0.02 | 100% |
| Yang_PhysRevE2008 | `kon2` | 1.15 | 0.56 | 0.82 | 0.53 | 0.70 | 16% |

What the baseline shows:

* **The problems are hard at this budget.** No method exceeds 80 percent on any problem.
  The positive-feedback switch (Lin_PhysRevE2016) and the six-rate promoter problem
  (Munsky_Science2012) defeat every method, and it is not for want of leverage: the failures
  land on the narrow directions (`p2`, `kon2`, `K`, `r1`) and on the six promoter rates,
  where the six-parameter landscape is flat in most directions at once. That is the
  benchmark's job: a set of problems every method solves would measure nothing.
* **CMA-ES's uncertainty handling (#661) helps or costs nothing.** With it on, success goes
  from 20 to 60 percent on Shahrezaei_PNAS2008 and from 0 to 40 percent on
  McKane_PhysRevLett2005, and is unchanged on Hlavacek_PNAS2001 (80 percent),
  Yang_PhysRevE2008 (20 percent), Lin_PhysRevE2016 and Munsky_Science2012 (0 percent). It
  also recovers `p2`, the narrow direction of McKane_PhysRevLett2005, twice as well as
  anything else (median 0.35 decades against 0.65 or worse).
* **Scatter search's noise handling (#660) is mixed.** It helps on Shahrezaei_PNAS2008 (40
  to 60 percent) and on the promoter rates of Munsky_Science2012 (median errors roughly
  halved), and it hurts on McKane_PhysRevLett2005 (20 to 0 percent) and Yang_PhysRevE2008
  (40 to 0 percent), where its median error on the narrow direction is worse than without
  it. Five seeds resolve a difference of about two fits in five, so those two drops are at
  the edge of what this run can tell apart; they are the first thing the full suite should
  settle.
* **No single method wins.** CMA-ES is the strongest on the receptor problem
  (Hlavacek_PNAS2001, 80 percent, 60 percent within 26 percent) and the predator-prey
  problem; differential evolution on the three-stage gene expression problem
  (Shahrezaei_PNAS2008, 80 percent); plain scatter search on the aggregation problem
  (Yang_PhysRevE2008, 40 percent).

## Scoring a change

Run the baseline methods again after the change with the same seeds, summarize, and compare
the success rates and median errors against `results/baseline_v1.md`. Five seeds resolve a
change in success rate of about two fits in five; a smaller effect needs more seeds
(`--seeds 10 --first-seed 1` reuses the first five).

## Adding a problem

1. Pick a published stochastic model with a citation; `BNGL-Models/models` is the pool.
2. Copy it to a new `problems/<id>/model.bngl`: bind each free parameter as `name name__FREE`,
   keep the observables that carry information, write one simulate action over a transient
   window, no seed, and record every adaptation in the header.
3. Write `problem.json` (copy a neighbour's): true values, bounds at least 0.3 decades from the
   truth on each side, identifiability, the grid, the budget.
4. Generate the data: `run_baseline.py generate --problems <id>`, and commit the `.exp`.
5. Run the default tests (they check the directory against itself) and the `recovery` tests
   (they check every identifiable parameter has leverage on the objective), and add the
   problem's rows to the leverage table with `run_baseline.py leverage --problems <id>`.
6. Run the baseline methods on it and add the rows to the results.

## Tests

`tests/test_stochastic_recovery_benchmark.py`. The default tier checks the frozen definitions,
the models and the committed data against each other and the scoring functions against the
protocol, without a backend. The `recovery` tier (`pytest -m recovery`) checks, through the
real backend, that every identifiable parameter has leverage on the objective, that data
generation is deterministic, and that a budgeted fit produces a complete record.

## Where this is going

The issue's plan is twenty to thirty problems, methods that are not ours, and a public home
with a permanent identifier. That home is [stochbench](https://github.com/wshlavacek/stochbench),
which holds the same six problems, the protocol as a document and as the same pure
`protocol.py`, and the results reported against them. The problems and `protocol.py` here
are the copy PyBNF's tests and runner read; `harness.py` and `run_baseline.py` are PyBNF's
runner for that collection, and depend only on the installed `pybnf` package.
