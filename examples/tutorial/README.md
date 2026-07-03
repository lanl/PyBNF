# PyBNF edition-2 tutorial

A hands-on tour of PyBNF's modern (**edition-2**) features, built on small ODE
models with known closed-form solutions. Every model is fit to synthetic data
generated *from that same model at known-true parameters*, so a correct fit
recovers the truth — which makes each lesson both a teaching example and an
automated regression test (see `tests/test_tutorial_examples.py`).

Each lesson is a self-contained folder: a commented model (`*.bngl`), its data
(`*.exp`), one or more heavily-commented fits (`*.conf`), and a short README.

## Getting started

You need [BioNetGen](https://bionetgen.org) (set `BNGPATH`) and PyBNF's bngsim
backend. Run any lesson from its own folder:

```bash
cd examples/tutorial/01_logistic_growth
pybnf -c logistic_growth_trf.conf
```

Results land in `output/` inside the lesson folder.

## The lessons

| # | Folder | You learn… | Feature(s) |
| --- | --- | --- | --- |
| 1 | [`01_logistic_growth`](01_logistic_growth) | your first fit; the edition-2 config surface | gradient least-squares (`trf`) |
| 2 | [`02_bateman_chain`](02_bateman_chain) | fitting several observables at once | differential evolution (`de`), multi-observable |
| 3 | [`03_gompertz_growth`](03_gompertz_growth) | global search then local polish | particle swarm (`pso`) + `refine` |
| 6 | [`06_step_input`](06_step_input) | when a gradient fit is *refused*, and why | gradient-refusal on non-smooth (`if()`) models |

*(More lessons — profile likelihood, bootstrapping, Bayesian sampling,
qualitative/BPSL constraints, per-observable noise, dose-response, and the PEtab
v2 round-trip — are being added; numbering leaves room for them.)*

## The edition-2 config surface, in one place

Every lesson uses these keys (see any `.conf` for the full commentary):

| Key | Meaning |
| --- | --- |
| `edition = 2` | opt into the modern config language |
| `model: file.bngl` | declare the model (no data bound here; no `begin actions` block) |
| `bngl_backend = bngsim` | simulate in-process with bngsim (required for gradient fits) |
| `job_type = …` | the algorithm: `trf`/`lbfgs` (gradient), `de`/`pso`/`ss`/… (metaheuristic), `am`/`dream`/… (Bayesian) |
| `objective = sos` \| `chi_sq` | the fit metric (`chi_sq` when the data has `_SD` columns) |
| `experiment: name, data: file.exp` | bind a named simulation to its data; the data's time column is the output grid |
| `uniform_var = p lo hi` | a free parameter, bound by name to model parameter `p`, searched in `[lo, hi]` |
| `refine = 1` | polish the best fit with a local optimizer |

## Not covered here (and why)

These models are **deterministic ODEs with closed-form solutions**, chosen so
every fit has a known right answer. That deliberately leaves three PyBNF
capabilities out of scope, because this palette can't exercise them honestly:

- **Stochastic simulation (SSA) and network-free (NFsim)** — these are tiny,
  fully-enumerable ODE networks; SSA would be non-deterministic and network-free
  simulation buys nothing.
- **Distributed-cluster execution** — orthogonal to the modelling; it's a
  deployment concern, not a per-model feature.

## Regenerating the data

The `.exp` files are the models' own output at the true parameters. To rebuild
them (needs bngsim + BNG2.pl):

```bash
python examples/tutorial/regenerate_data.py            # all lessons
python examples/tutorial/regenerate_data.py 02_bateman_chain   # one lesson
```
