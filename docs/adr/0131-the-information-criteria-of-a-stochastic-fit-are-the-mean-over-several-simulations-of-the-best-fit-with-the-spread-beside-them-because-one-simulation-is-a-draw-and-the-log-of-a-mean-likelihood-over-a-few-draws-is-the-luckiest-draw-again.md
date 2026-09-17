# The information criteria of a stochastic fit are the mean over several simulations of the best fit, with the spread beside them, because one simulation is a draw and the log of a mean likelihood over a few draws is the luckiest draw again (issue #676)

## Status

Accepted. Completes #659, the best-fit confirmation stage, for the one end-of-fit artifact
that stage left on a single simulation.

Amended by #741 (2026-09-17): the file gained a `replicates_requested` line beside
`replicates`, and says in words how many runs produced nothing when the two differ. The
example and the sentence about dropped runs below describe the file as this ADR shipped it;
nothing about the decision changed, only what the file discloses about it. See ADR-0146.

## The defect

`Results/information_criteria.txt` reports AIC, BIC and AICc from the full normalized
log-likelihood of the best fit. That log-likelihood came from re-simulating the best fit
exactly once, in `Algorithm._compute_information_criteria`, because an optimizer discards
simulation data after scoring it on the workers and this is the only place the best fit's
data is in hand.

For a stochastic model one simulation is a draw. So the reported log-likelihood was a noisy
number, and the AIC built on it moved from one run to the next for the same parameter set by
an amount that had nothing to do with the model. A difference of two in AIC is treated as
meaningful, and nothing in the file said whether the simulation noise was larger than that.

Two things made it worse than noisy.

* Under the default `stochastic_seed = auto` policy the seed is derived from the parameter
  values and the replicate index (`pybnf/_seed.py`), and the information-criteria job carried
  index 0. So the re-simulation reproduced the trajectory the search itself had scored for
  that parameter set. Before #659 that was the luckiest draw of the whole run, for the reasons
  #659 gives. After #659 it is the confirmed winner's search draw, which is less biased but is
  still one draw.
* After #659 a run writes two files about the same parameter set: `best_fit_confirmation.txt`
  with an objective value averaged over `best_fit_replicates` runs and its standard error, and
  `information_criteria.txt` with a log-likelihood from one draw. They visibly disagreed, and
  nothing said which to trust.

The same single-draw path writes `information_criteria_backup.txt` at every checkpoint (#560).

## The decision

At the end of a stochastic fit the best fit is simulated `best_fit_replicates` times for the
information criteria, at replicate indices past every index the fit and the confirmation
stage used, so each is a fresh draw. The log-likelihood reported is the **mean of the
per-simulation log-likelihoods**, and the file says how many runs that was and how far they
spread:

```
k	23
n	120
replicates	10
log_likelihood	-51.20409
log_likelihood_standard_error	0.8312
AIC	148.4082
BIC	212.5231
AICc	159.9082
```

AIC, BIC and AICc are computed from the mean, and each carries twice the standard error,
since each is `-2 lnL` plus a constant. The comment block above the values says so, and says
that two models whose AIC values differ by less than that have not been told apart.

Everything else keeps its single simulation at index 0, and its file says `replicates 1` and
`n/a (one simulation)` so a reader can tell: a deterministic fit, whose simulation is exact;
a stochastic fit whose seed policy pins every model to one trajectory
(`_replicates_would_differ`); a legacy-edition fit, where `best_fit_replicates` defaults to
off so a conf that names no edition keeps costing what it always has (ADR-0031); a fit whose
wall-time budget is spent, by the confirmation stage's argument that a budget is a promise
about the whole run; and every checkpoint, which fires on a cadence and is cheap by
construction (#560).

The simulations go out through the dask client together, as the confirmation stage's do,
with no calculator attached so the simulation data comes back to be scored in-process
through the same normalize, postprocess, pointwise-`log_density` path as before. Without a
client they run one after another. A run that fails or scores nothing is left out and
`replicates` reports the number used; runs that scored a different number of points than the
rest are left out too, since a sum over a different `n` is a different quantity. A profiled
noise scale (ADR-0108) is averaged over the same runs.

That last rule is the one #741 came back to. Reporting only the number used left an average
over 3 of 10 runs reading exactly like an average over 3 of 3, in the file whose whole
purpose is a comparison between two such averages; the fix reports the number run beside it
and says what was lost. The averaging itself is unchanged, and no minimum-success rule was
added here -- #720's threshold exists because candidate means were ranked against each other,
and there is no competing candidate in this file.

## Why the mean of the log-likelihoods, and not the log of the mean likelihood

#676 says this has to be decided before any code, and that the two numbers answer different
questions. They do.

For a model whose simulator is stochastic the likelihood of the data is the marginal over
trajectories, `p(d | θ) = E_traj[ p(d | traj, θ) ]`. The estimate of it from `m` simulations
is `log(mean_i L_i)`, the log of the mean likelihood. The mean of the log-likelihoods,
`mean_i log L_i`, is a lower bound on it by Jensen's inequality, and the gap between them is
set by the trajectory-to-trajectory spread.

Three reasons the mean of the logs is the number to report.

1. **It is what the confirmation stage already averages.** For a likelihood objective the
   objective value is the reduced negative log-likelihood, and `best_fit_confirmation.txt`
   reports its mean over the replicates. Reporting the mean log-likelihood here means the two
   files estimate the same quantity for the same parameter set, which is the disagreement
   #676 asks to remove. Reporting the log of the mean likelihood would replace one
   disagreement with another, this time between two numbers that are both "the average".

2. **The log of the mean over a few draws is the luckiest draw again.** `log mean_i exp(l_i)`
   is a log-sum-exp. When the spread of the `l_i` is more than a nat or two, which is
   exactly the regime where any of this matters, it is the maximum plus a small correction,
   so a ten-draw estimate of it is decided by the single best simulation. That is the "best
   of many noisy draws" optimism #659 removed, back again over ten draws instead of a
   hundred thousand, with far higher variance than the mean has, and no honest standard
   error can be put on it from ten values.

3. **The gap is visible.** The standard error the file reports is a direct measure of the
   spread that separates the two quantities. With a spread of a tenth of a nat they agree
   to a rounding error. With a spread of five nats neither the AIC nor the comparison it
   feeds should be trusted from ten simulations, and the file says so in the only way that
   helps, which is with a number.

The consequence to be clear about: the reported AIC of a stochastic fit is built on a lower
bound on the log marginal likelihood, not on the marginal itself. Comparing two stochastic
models on it compares those bounds. For models whose trajectory spread is similar that is
the same comparison; for models whose spread differs a lot it is not, and the two standard
errors are what shows it. A proper likelihood for a stochastic model is what #665, the
synthetic-likelihood investigation, is about; if that lands, it replaces this estimator
rather than refining it.

## Cost

`best_fit_replicates` more simulations of one parameter set at the end of a stochastic fit
with a likelihood objective, ten by default, all submitted at once. Next to the confirmation
stage's `best_fit_candidates` times `best_fit_replicates`, a hundred by default, that is a
tenth more end-of-run work, and no more wall clock on a machine with ten free processors.
The checkpoint's cost is unchanged.

## Consequences

* `Results/information_criteria.txt` and its checkpoint gain two key/value lines,
  `replicates` and `log_likelihood_standard_error`, in every fit. A deterministic fit's file
  says `1` and `n/a (one simulation)`, and its other lines are byte-identical to before. One
  parser still reads either file.
* `pybnf.objective.InformationCriteria` gains the two fields, with defaults, and
  `replicated_information_criteria` builds one from several log-likelihoods.
* `Algorithm._compute_information_criteria` takes `replicates` and `client`;
  `_information_criteria_replicates` decides the count by the confirmation stage's rules, and
  `_run_information_criteria_jobs` runs the simulations through the client or in-process.
* `best_fit_replicates` now governs two things, both "how many times the best fit is run
  again at the end of a stochastic fit". Its documentation says so.
* What #676 asked for and this does not do is measure how large the effect is on a real
  model. The two new lines are that measurement, taken by every stochastic fit from now on.

## Verification

`tests/test_information_criteria_replicates.py`: the arithmetic; which replicate indices run,
in-process and through a client; a run that cannot be scored, one that raises, one with a
non-finite log-likelihood, and one that scored a different number of points; the averaged
profiled noise scale; the two new lines and the console line; the rules that decide the
count; and that the checkpoint still asks for one simulation. The existing information-
criteria, checkpoint, noise-profiling and confirmation tests pass unchanged, apart from one
end-of-fit stub that now accepts the new keyword arguments.
