# A best-fit candidate must produce a usable objective value in more than half of its replicate runs to win the confirmation ranking, because an average over only the runs that worked is the lucky draw the stage exists to remove (issue #720)

## Status

Accepted and implemented (2026-09-17), under every edition that runs the stage at all. It
changes which parameter set a run reports only for a fit in which a candidate failed half of
its replicate runs or more. `best_fit_confirmation.txt` gains columns and lines on every such
fit, failures or not.

## The problem

The best-fit confirmation stage (#659) exists because a stochastic model's objective value is
a noisy measurement. A fit picks its answer by taking the best value it ever saw over tens of
thousands of single simulations, so the winner of that comparison is very often the parameter
set that got a lucky simulation rather than the one that is genuinely best. The stage takes
the search's top `best_fit_candidates` parameter sets, runs each of them `best_fit_replicates`
more times, ranks them by their average, and pins the winner as the run's best fit.

A replicate run does not always produce a number. `Algorithm._run_confirmation_replicates`
counts a run as failed and drops it when the simulation is a `FailedSimulation` — a crash, or
a `wall_time_sim` timeout, both genuinely intermittent for SSA and NFsim — and when the
objective value it scores is `None` or not finite, which a stochastic trajectory that hits
zero counts reaches easily under a log-based objective.

`mean_objective` averaged what was left. The failure count travelled on the row and was
printed in a column, but it entered no comparison: `ranked` keyed on that mean alone, and
`winner` disqualified a candidate only when every single one of its runs had failed. The
ranking that decided the run's answer was therefore a survivor-only mean over a sample size
that varied from candidate to candidate, anywhere from 1 to `best_fit_replicates`.

That is the very thing the stage was built to prevent, reached through a different channel. A
parameter set that fails nine runs of ten and survives one is scored on exactly one
simulation — the max-of-lucky-draws estimate again, now with the unlucky draws deleted rather
than averaged in — and it beat a parameter set measured honestly over ten. Run end to end on
the test harness, a candidate returning a non-finite value on nine of ten runs and 5.0 on the
tenth won against one returning 5.1 on all ten, and `_emit_best_fit_confirmation` pinned it,
which is by design how the saved simulations, the best-fit BNGL, the information criteria, a
refine's start point and a bootstrap replicate's answer all come to agree. The expected
objective value of the winner was effectively infinite; of the loser, 5.1.

It was also silent where it mattered. `console_lines` never mentioned failures, and with a
single survivor `standard_error` is `None`, so the documented "raise `best_fit_replicates` if
the standard errors overlap" guidance could not flag it either. The console's one line about a
changed answer, "the one the search liked best does worse when it is run again", fires
whenever the winner is not the search's own top pick, and says nothing about the average it is
comparing being over one run. Turn the same situation around, so that the *flaky* candidate is
the search's top pick and a reliable one displaces it, and the sentence is simply false: the
search's pick had the better average and lost on reliability, which is not what "does worse
when it is run again" tells the reader. The only warning in the stage fires when *no* candidate
has a usable score.

PyBNF's other replicate-averaging path takes the opposite line. `JobGroup.job_finished` and
`average_results` turn a whole smoothing group into a `FailedSimulation` the moment one
sub-run fails, rather than averaging the survivors, so survivor-only averaging was never a
house-wide rule this stage was following.

## The decision

### A failure is an outcome, not a missing measurement

This is the whole of it. A replicate that fails, or that scores something that is not a finite
number, is not a datum that went astray. It is a result, and a bad one: the parameter set
could not be simulated, or it was simulated and scored worse than any number. Dropping it
biases the estimate downward, and the size of the bias grows with how often it happens.

Averaging only the runs that produced a value estimates a candidate's objective value *given
that the run worked*. That is worth quoting when working is the normal case and is not when it
is the exception, and nothing in the old code marked the difference.

### The bar: more than half of the runs that came back

`best_fit_confirmation.confirmed` accepts a candidate when more than half of the replicate runs
that came back with a verdict produced a usable value. `ranked` sorts confirmed candidates
first, by average, and unconfirmed ones after all of them, also by average; `winner` returns
`None` unless the leader is confirmed.

Half is not a tuning knob picked for its round number. It is exactly the condition for the
middle of all of a candidate's runs to be a real number. A failure is known to be worse than
every finite value, so a candidate's runs are a sample in which some values are known only as
"worse than all the rest"; that sample has a finite median precisely when more than half of it
is finite. A candidate that fails half of its runs or more has no finite middle at all, and
its average over the survivors is describing the minority that happened to work. Above the
bar, the survivor mean is an average over most of what happened; below it, it is a lucky-draw
estimate wearing an average's clothes.

The bar is written against the runs that came back (`attempts`, successes plus failures)
rather than against `best_fit_replicates`. A replicate that was cancelled rather than run is
the cluster's doing and not the parameter set's, and counting it against the candidate would
punish it for something it did not do.

### Nothing is hidden, and a candidate that loses a few runs still competes

An unconfirmed candidate keeps its row, with the average it reached and the number of runs it
reached it over. `mean_objective` is unchanged and still reports that average, so the number a
reader may want to see is still there; what changed is that it can no longer decide the
answer. A new `confirmed` column marks each row, an `unconfirmed` line names the demoted
candidates, and `winner_runs` and `winner_failed` put the winner's own sample size in the file.

A candidate that loses a minority of its runs is affected by none of this. It is confirmed, it
is ranked on its average against the other confirmed candidates exactly as before, and it can
win — with a note above the winner block saying how many runs it lost, because a parameter set
that cannot be simulated every time is worth looking into whether or not it is the answer.

### When nobody clears the bar, the stage pins nothing

If no candidate is confirmed, `winner` returns `None`, which is the path the stage already had
for "no candidate produced a usable value at all": the search's own pick stands, nothing is
pinned, and the report says so. The text distinguishes the two cases, because "nothing could be
run" and "everything ran and mostly failed" call for different next steps, and the second is a
statement about the model and the cluster that the reader should act on.

### The console stops asserting a thing that is not true

`console_lines` now reports the winner's failed runs, how many candidates were not confirmed,
and, where the winner is not the search's own pick, which of the two reasons applies: the
search's pick ran worse, or the search's pick was not confirmed. The second was previously
reported as the first.

## Alternatives

* **Leave it as it is, and only disclose.** The issue's own floor: at the very least say that
  nine replicates produced nothing. Disclosure is necessary and is implemented, but it is not
  sufficient. The file is read after the fact, while the pinned parameter set has already
  propagated into every downstream artifact, and a reader who has to notice a failure column
  to distrust the headline number is being asked to do the ranking's job. Adopted as half of
  the fix, rejected as the whole of it.
* **Count a failure as an infinite objective value,** so any failure at all sinks a candidate.
  It is the honest reading of "the expected objective value is infinite", and it is what
  `JobGroup` does for a smoothing group. Rejected: at ten replicates of a stochastic model on
  a cluster, one transient timeout is common and is often not the parameter set's fault, and
  this would throw away a good candidate for it. It also contradicts the stage's own stated
  intent, pinned in a test since #659, that one bad draw does not kill a good candidate.
* **Rank by failure count first, then by average.** Rejected as lexicographic: it makes a
  single failure worse than any difference in objective value, so a candidate with ten runs of
  ten at an objective of 500 would beat one with nine of ten at 1.
* **Scale the average by the success rate,** or otherwise penalize continuously. Rejected on
  arithmetic before policy: PyBNF objective values are not sign-constrained — a likelihood
  objective is routinely negative — so a multiplicative penalty improves a negative score and
  any additive one needs a scale the stage does not have.
* **Make the bar a configuration key.** Rejected. It pushes a statistical judgement onto the
  user at the moment they are least able to make it, and the median argument gives the
  threshold a meaning that an arbitrary number would not have. A user who wants the old
  behavior has `best_fit_replicates = 0`.
* **Fall back to the best unconfirmed candidate when none is confirmed,** rather than pinning
  nothing. Rejected: pinning a parameter set that fails most of its runs is exactly the harm
  this ADR removes, and it is no less harmful when every candidate is bad. Falling back to the
  search's pick does not claim a confirmation that did not happen, and the table still shows
  every number.
* **Rank on the median of all the runs** rather than using the median only to set the bar. It
  is the estimator the censoring argument actually points at, and it would penalize a
  candidate that fails a few runs continuously instead of at a threshold. Rejected for blast
  radius, not for correctness: `mean_objective` is the number the docs teach, the number
  `pin_best` records, and the number `information_criteria.txt` is deliberately aligned with
  (ADR-0131), and switching to a median would change the reported value of every stochastic
  fit, including every fit with no failures at all. A bug fix should not do that.

## Consequences

* A fit in which no candidate failed half of its runs or more reports the same winner and the
  same objective value as before. Only the report gains columns.
* A fit in which one did now reports a different winner, and one that is measured rather than
  lucky. Runs whose reported best fit changes are the runs the issue is about.
* `best_fit_confirmation.txt` gains a `confirmed` column, `winner_runs` and `winner_failed`
  lines, and an `unconfirmed` line when it applies. The file is human-readable and nothing in
  the tree parses it, but a user's own script that reads the table by column index will see
  `confirmed` inserted before `search_objective`.
* The information criteria still drop failed replicates when averaging the log-likelihood of
  the one pinned parameter set. That is an estimate of a single parameter set rather than a
  ranking, so the selection argument above does not apply to it in the same way; with the
  winner now required to be reliable, the parameter set it describes is one whose runs mostly
  work. The averaging is left as it is. What this ADR missed is that the sibling file did not
  even *say* it had dropped anything, so the two files stopped reporting the same kind of
  measurement the same way the moment this change landed; #741 closed that gap by reporting
  the count of runs made beside the count used. The threshold argument above still does not
  transfer, and no minimum-success rule was added there.
