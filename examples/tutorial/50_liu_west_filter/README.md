# Lesson 50 — Forecasting counts as they arrive (the Liu–West filter)

**Feature:** `job_type = lwf`, the Liu–West filter, and its `lwf_*` keys · **Difficulty:** ★★★ · **Tier:** default (CI runs a short version of this job)

Surveillance counts arrive one day or week at a time, and what is wanted after each
is a forecast of the next few. The **Liu–West filter** (`job_type = lwf`) takes the
rows one at a time instead of refitting the series. Each of its particles holds the
free parameters, its own model state and a weight. Before each row the kernel moves
every particle's free parameters; each particle's state is integrated over the
interval and weighed by the likelihood of the count, and uneven weights are
resampled. After the last row it forecasts the next few intervals.

Its output is a **forecasting sample** of a model whose free parameters drift by the
kernel at every row, **not** the posterior of fixed parameters that `am`
([lesson 39](../39_adaptive_mcmc)) samples. Every output file says so, and the
Liu–West filter section of the [algorithms page](../../../docs/algorithms.rst)
explains why.

## The model and data

An SIR outbreak ([`sir_outbreak.bngl`](sir_outbreak.bngl)) in 50,000 people, in
which each infection also adds one `C()`, so the observable `cases` counts the
infections since t = 0. `gamma` and `N` are fixed; the free parameters are `beta`,
the number initially infectious `I0`, and the dispersion `r` of the count noise.

[`cases.exp`](cases.exp) holds 42 daily counts of new infections, drawn by
[`regenerate_data.py`](regenerate_data.py) (negative binomial, dispersion 20, seed 2026) around
the model's daily increase of `cases` at the model file's values. The script
integrates the model by hand, so it needs only numpy:

```bash
python regenerate_data.py
```

The rows must be one step apart, the first one step after t = 0; a missing interval
is a row holding `nan`, which the filter integrates over without scoring.

## The configuration

[`liu_west_filter.conf`](liu_west_filter.conf) is commented line by line. The count
is declared by two `noise_model` lines with the same dispersion:

```
noise_model = neg_bin, dispersion = fit r, location = mean
noise_model cases = neg_bin, dispersion = fit r, location = mean, cumulative
```

The first is the whole-fit objective edition 2 requires; the second adds the
`cumulative` flag of [lesson 28](../28_cumulative_counts), so each row is scored
against the model's increase of `cases` over the interval. The kernel moves each
parameter `x` (here the logit of its position in its box) to

```
x' = a x + (1 - a) m + h (L z),      a = sqrt(1 - h^2)
```

with `m` and `L L^T` the population's weighted mean and covariance and `z` standard
normal. `lwf_jitter` sets `h`, and `lwf_parameter_jitter = r 0.05` gives `r` its
own. `h` says how fast the free parameters may drift, so the forecasts depend on it
however many particles you use. `lwf_forecast_intervals = 7` forecasts a week.

## Run it

```bash
pybnf -c liu_west_filter.conf
```

The console prints a summary (the rows assimilated, the moved parameters and their
`h`, the weight ESS, the log evidence of the drifting-parameter model, the forecast)
and what the output is. Under `output/liu_west_filter/Results/LWF/`:

| File | Holds |
| --- | --- |
| `parameters.txt` | an equal-weight resample of the population after the last row, in each parameter's own units |
| `parameters_by_row.txt` | each parameter's weighted mean and 2.5, 50 and 97.5 percent quantiles after each row |
| `predicted_counts.txt` | the expected count at each row along each written particle's ancestral line |
| `weight_ess.txt` | per row: the weight ESS, whether it resampled, the distinct, failed and zero-increment particles, and the log evidence |
| `forecasts.txt` | a forecast count per written particle for each of days 43 to 49; a column's quantiles are that day's forecast interval |
| `summary.txt` | the run summary |

The state file `lwf_state.npz` sits beside `Results/`. `I0` acts only through the
initial state, so the filter never moves it, and the summary says so.

## Things to try

- **Change `h`.** Set `lwf_jitter = 0.05`, then `0.3`, and compare
  `parameters_by_row.txt`, `forecasts.txt` and the log evidence, which belongs to
  the drifting-parameter model and so changes with `h`.
- **Independent runs.** Add `lwf_independent_runs = 4`. Each run filters the data
  from its own prior draw, on a local cluster PyBNF starts, and writes its files to
  `Results/LWF/run_<r>/`; the files in `Results/LWF/` stack the four with equal weight.
- **A new row arrives.** Remove the last two rows of `cases.exp` and run. Put them
  back, add `lwf_continue = 1`, and run again with `-o`. The second run assimilates
  only the two new rows and writes, byte for byte, what a run over all 42 writes. A
  continuation is refused if the model, priors, noise model, kernel settings,
  particles, seed or an assimilated row changed.

[`tests/test_tutorial_liu_west.py`](../../../tests/test_tutorial_liu_west.py) runs a
short version of this job and of its continuation in CI.
