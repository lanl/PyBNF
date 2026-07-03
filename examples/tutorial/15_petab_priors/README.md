# Lesson 15 — Priors in PEtab (declaring what you believe)

**Feature:** PEtab v2 `priorDistribution` / `priorParameters` → PyBNF `FreeParameter` priors · **Difficulty:** ★★☆

Lesson 13 showed the linter catching a *malformed* prior (`bad_prior`). This is the
positive side: a PEtab problem where every prior is **well-formed**, and each one
imports into a PyBNF fit as the free-parameter family it names. Priors are how you
tell the fit what you believe about a parameter *before* it sees the data — a
handle you'll use for real in the Bayesian lesson, and a regularizer that keeps an
optimizer honest.

The problem here is a mass-action **receptor–ligand binding** model,
`L + R ⇌ C` ([`binding.bngl`](binding.bngl)), with four estimated parameters. The
whole lesson lives in one table — [`parameters.tsv`](parameters.tsv):

| Parameter | What it is | `priorDistribution` | `priorParameters` | Why this family |
| --- | --- | --- | --- | --- |
| `kon` | association rate | `log-normal` | `0;1` | a rate spanning **orders of magnitude** — belief lives on a log scale |
| `koff` | dissociation rate | `gamma` | `2;0.5` | strictly **positive**; a shape/scale prior on `(0, ∞)` |
| `R0` | total receptor | `normal` | `30;5` | a quantity **measured with error** — roughly Gaussian |
| `L0` | ligand dose | *(blank)* | *(blank)* | a dose **you set yourself**; you know only the range |

Reading that top to bottom *is* the lesson: the prior family is a statement about
what kind of thing the parameter is. A blank prior (the `L0` dose) is PEtab's way
of saying "no explicit prior" — the validator defaults it to a **uniform over the
bounds**.

## How each prior imports

PyBNF maps every PEtab `priorDistribution` to one of its native `*_var`
free-parameter families (the same families the `.conf` grammar uses). Run the
import and look at the generated job:

```python
from pybnf.petab import import_job
import_job("problem.yaml", "imported/", job_type="de")   # writes imported/imported.conf
```

The four rows above come back as (`imported/imported.conf`):

```
parameter: kon, prior: normal, parameter_scale: log10, mean: 0, sd: 0.4342944819, lower: 0.001, upper: 10
parameter: koff, prior: gamma, shape: 2, scale: 0.5, lower: 0.001, upper: 10
parameter: R0, prior: normal, mean: 30, sd: 5, lower: 1, upper: 100
uniform_var = L0 1 100
```

Three things worth noticing:

- **A bounded prior becomes a `parameter:` record.** The truncated families
  (`kon`, `koff`, `R0`) carry `lower`/`upper` walls, so they import as the new-era
  `parameter:` record — the only grammar with room for a truncation box. The
  untruncated `L0` is the terse positional `uniform_var` line.
- **`log-normal` → `parameter_scale: log10`.** PyBNF parameterizes its log families
  in log₁₀, while PEtab states log-normal parameters in *natural* log — so the
  import converts (the `1` in `priorParameters` becomes `1/ln 10 ≈ 0.4343`). The
  distribution over the parameter is unchanged; only the sampling coordinate
  differs.
- **The bounds truncate the prior.** `[lowerBound, upperBound]` become reflecting
  walls on the density, not just search limits.

Then run it like any other job:

```bash
pybnf -c imported/imported.conf
```

## The full prior catalog

The gallery above uses four families; PyBNF supports the whole PEtab v2 catalog,
each mapping to a `*_var` keyword (and a `log…_var` / `ln…_var` scale variant):

| PEtab `priorDistribution` | PyBNF family | Parameters |
| --- | --- | --- |
| `uniform` | `uniform_var` | lower, upper |
| `normal` | `normal_var` | mean, sd |
| `laplace` | `laplace_var` | location, scale |
| `log-uniform` | `loguniform_var` | lower, upper |
| `log-normal` | `lognormal_var` | mean, sd *(natural log)* |
| `log-laplace` | `loglaplace_var` | location, scale *(natural log)* |
| `cauchy` | `cauchy_var` | location, scale |
| `gamma` | `gamma_var` | shape, scale |
| `exponential` | `exponential_var` | scale |
| `chisquare` | `chisquare_var` | degrees of freedom |
| `rayleigh` | `rayleigh_var` | scale |

Whatever the source, finite `lowerBound`/`upperBound` truncate the family to a
reflecting box (two walls → a two-sided box; one wall + an open side → a
half-bounded box).

## Where priors earn their keep

A prior is inert until an algorithm uses it. An optimizer (`de`, `trf`, …) treats
it as a regularizing penalty that pulls implausible parameters back; a Bayesian
sampler treats it as the actual prior of Bayes' rule and returns a posterior. This
lesson establishes the *vocabulary*; the Bayesian lesson puts it to work.

## Regenerating the fixture

The committed problem is produced by a small dev tool; the parameters table is
written straight from the test manifest so the fixture and the expected import can
never drift:

```bash
python regenerate_fixtures.py
```

The expected import for each parameter lives test-side in
[`_manifest.py`](../_manifest.py) (`PRIOR_CASES`), checked by
[`tests/test_tutorial_priors.py`](../../../tests/test_tutorial_priors.py): the
problem lints clean, and every row maps to the recorded `FreeParameter`.

## See also

- **Lesson 13** ([`13_petab_lint_clinic`](../13_petab_lint_clinic)) — the linter
  catching a *malformed* prior (`bad_prior`).
- **Lesson 12** ([`12_petab_roundtrip`](../12_petab_roundtrip)) — exporting a PyBNF
  job *to* PEtab (priors travel the other way too).
