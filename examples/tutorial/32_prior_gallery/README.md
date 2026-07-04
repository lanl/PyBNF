# Lesson 32 — The prior-family gallery

**Feature:** the full catalog of prior families (`normal_var`, `laplace_var`, `gamma_var`, `beta_var`, `half_normal_var`, … and `student_t` via the `parameter:` record) · **Difficulty:** ★★★ · **Tier:** slow

[Lesson 27](../27_priors) made the case for an informative prior with a single
family (gamma). This lesson is the **gallery**: it holds the model, the data, and
the sampler fixed and swaps *only* `k2`'s prior line across the whole catalog of
families PyBNF ships, so you can see how each one is spelled, what shape of belief
it encodes, and how each one sharpens the same weakly-identified parameter.

> **The one rule, again:** a prior acts **only inside a Bayesian sampler**
> (`dream` / `mh` / `pt`). A point optimizer (`de`, `pso`, `trf`, …) maximizes the
> likelihood alone and never looks at the prior family. So this whole lesson is a
> sampler — see [Lesson 27](../27_priors) for why.

## The setup (identical to Lesson 27)

The same [Bateman chain](bateman_chain.bngl) `A → B → C`, fit to two channels of
very different quality:

| channel | `_SD` | constrains |
|---------|-------|------------|
| `Obs_A` | 3  (tight) | `A(t) = A₀e^{-k1 t}` → pins `k1` hard |
| `Obs_C` | 25 (loose) | the only, noisy, handle on `k2` |

So `k1` is well identified and `k2` is **weakly** identified — the parameter a
prior can actually help. Every conf below leaves `k1` a flat `uniform_var` and
changes only `k2`.

## Every family has a keyword

PyBNF ships a `*_var` keyword for **every** prior family, and the positional
numbers after it are that family's *own* parameters. Each conf here encodes the
same independent estimate — *"an earlier experiment measured `k2 ≈ 0.25 ± 0.05`"* —
but in a different family, so you can see that the family is a **modeling choice**
about the *shape* and *support* of that belief:

| conf | line | support | what the family says |
|------|------|---------|----------------------|
| [`flat_prior`](flat_prior.conf) | `uniform_var = k2 0.02 2.0` | `[0.02, 2.0]` | flat — anything in range, equally (the control) |
| [`normal_prior`](normal_prior.conf) | `normal_var = k2 0.25 0.05` | ℝ | symmetric bell, `mean` ± `sd` — the textbook estimate |
| [`laplace_prior`](laplace_prior.conf) | `laplace_var = k2 0.25 0.0354` | ℝ | like normal but **heavier-tailed** (more forgiving of prior↔data conflict) |
| [`gamma_prior`](gamma_prior.conf) | `gamma_var = k2 25 0.01` | `(0, ∞)` | positive & right-skewed — the natural prior for a **rate** (`mean = shape·scale`) |
| [`beta_prior`](beta_prior.conf) | `beta_var = k2 18.5 55.5` | `[0, 1]` | bounded — the prior for a **fraction / probability** (`mean = α/(α+β)`) |
| [`half_normal_prior`](half_normal_prior.conf) | `half_normal_var = k2 0.313` | `(0, ∞)` | one parameter; **weakly**-informative "small & positive" (mode at 0) |
| [`student_t_prior`](student_t_prior.conf) | `parameter: k2, prior: student_t, …` | ℝ | heavy-tailed **robust** normal; three parameters → record only |

`normal`, `laplace`, `gamma`, and (in a bounded way) `beta` are all tuned to the
same `0.25 ± 0.05` belief — they differ in *support* (can it go negative? above 1?)
and *tail weight*, not in where they sit. The last two are different in kind:

- **`half_normal`** is a *one-parameter* family (a single positional number), and
  its mode is at **0**, not at `0.25`. It doesn't say "`k2` is about 0.25" so much
  as "`k2` is a small positive number, probably below ~0.6" — a gentler,
  weakly-informative statement. It still narrows the flat posterior, but less than
  the sharply-centered families. The other one-parameter families read the same
  way: `exponential_var = k s`, `half_cauchy_var = k s`, `chisquare_var = k df`,
  `rayleigh_var = k s`.
- **`student_t`** has **three** parameters (`df`, `location`, `scale`), so it can't
  fit the two-number positional grammar. It is authored through the new-era
  **`parameter:` record**, where every field is named — the general way to declare
  *any* prior, and the *only* way to reach a ≥3-parameter family:

  ```
  parameter: k2, prior: student_t, df: 4, location: 0.25, scale: 0.05, lower: 0, upper: 2
  ```

  The `df` knob dials tail weight (small `df` = fat tails = tolerant of a
  surprising value; `df → ∞` = an ordinary normal); `lower`/`upper` truncate it to
  positive rates. The same record spells the two-number families too, if you prefer
  named fields: `parameter: k2, prior: normal, mean: 0.25, sd: 0.05`.

A `log`-prefixed twin (`lognormal_var`, `loglaplace_var`, `loggamma_var`, …) places
any of these families in `log10` space — natural for a parameter that ranges over
orders of magnitude.

## Run the gallery and compare `k2`

```bash
pybnf -c flat_prior.conf
pybnf -c normal_prior.conf        # ... and laplace_, gamma_, beta_,
pybnf -c student_t_prior.conf     #     half_normal_ in between
```

Then compare `k2` across the `Results/credible95.0_final.txt` files:

- **`k2` narrows under every informative family.** The flat prior leaves `k2`'s
  95% credible interval wide (the noisy `Obs_C` is all it has); each informative
  family collapses it onto a tighter interval that still brackets the true `0.25`.
  The sharply-centered families (`normal`, `gamma`, `laplace`, `student_t`,
  `beta`) narrow it the most; the weakly-informative `half_normal` narrows it
  least — exactly what its mode-at-0 shape predicts.
- **`k1` never moves.** The precise `Obs_A` pins `k1` in *every* run — strong data
  overrides any prior. A prior buys leverage only where the data is weak.

That is what [`tests/test_tutorial_prior_gallery.py`](../../../tests/test_tutorial_prior_gallery.py)
checks: a fast **structural** pass builds every conf and confirms each `k2` line
yields the family its filename names (positional numbers mapped to the family's
parameters, density oracled against `scipy`); a **slow** pass samples a
representative trio (`normal`, `gamma`, `student_t`) and asserts each 95% interval
is clearly narrower than the flat one and brackets the truth, with `k1` bracketed
in every run.

## Where this sits

- [Lesson 27](../27_priors) — the single-family (gamma) version of this idea, with
  the flat-vs-informative contrast spelled out.
- [Lesson 15](../15_petab_priors) — the *same* families imported from a PEtab
  `priorDistribution` column (priors declared in a standard exchange format).
- [Lesson 17](../17_bayesian_uncertainty) — the DREAM sampler and credible
  intervals, introduced with flat priors.
- [Lesson 26](../26_mcmc_samplers) — the other samplers (`mh`, `pt`) that honor
  priors the same way DREAM does.
- The `parameter:` record used for `student_t` is the general, fully-labeled
  free-parameter syntax — see the edition-2 surface table in the
  [tutorial README](../README.md).

## Regenerating the data

```bash
python ../regenerate_data.py 32_prior_gallery
```

The `.exp` is the Bateman model's own output at the true rates, with the same
per-observable `_SD` column as Lesson 27 (tight `Obs_A`, loose `Obs_C`). Nothing
about the prior enters the data; the prior lives only in the confs.
