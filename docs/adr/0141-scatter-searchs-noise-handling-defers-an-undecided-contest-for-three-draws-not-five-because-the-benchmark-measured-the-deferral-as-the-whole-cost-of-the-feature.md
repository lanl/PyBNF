# Scatter search's noise handling defers an undecided contest for three draws, not five, because the benchmark measured the deferral as the whole cost of the feature (issue #663; amends ADR-0136)

## Status

Accepted and implemented (2026-09-11). Amends ADR-0136, whose `ss_noise_max_draws` default
of 5 this replaces. The evidence is in the stochastic recovery benchmark
(`benchmarks/stochastic_recovery/`, ADR-0140) and its README.

## The problem

ADR-0136 gave scatter search a noise-aware reference set: a parent-versus-child decision the
noise leaves in doubt is not made, both sides are drawn again, and the contest waits, up to
`ss_noise_max_draws` draws each, after which the means decide. The default was five, chosen
by reasoning rather than measurement, as the whole of #660 was: nothing existed then to
measure it with.

The benchmark, once it existed, measured it. Over twenty seeds on four problems, seed-paired
so both variants start from the same population, the noise handling at its default changed
the success rate on no problem and left the final parameter error worse in 48 of 80 paired
seeds (sign test p = 0.09). Instrumenting two runs showed why. With the deferral at five, the
search accepted about half as many replacements into its reference set (60 and 45 against 91
and 105), left about a hundred contests per run undecided for another round, spent a tenth of
its evaluations re-drawing points it had already seen, and completed two fewer rounds. While
a contest is open the old parent keeps seeding the next round's combinations. The reference
set evolved half as fast, and at a fixed budget that cost cancelled the benefit of the better
ranking.

## The decision

The deferral shortens from five draws to three. Measured on the same twenty seeds and four
problems, pooled over 80 paired seeds:

| setting | successes of 80 | median final error (decades) | mean final error |
|---|---:|---:|---:|
| plain scatter search | 17 | 0.600 | 0.621 |
| deferral 5 (the old default) | 18 | 0.631 | 0.720 |
| deferral 2 | 18 | 0.549 | 0.648 |
| deferral 3 | 24 | 0.674 | 0.682 |

* A deferral of 2 beats 5 on the final error in 50 of 79 decided seeds (p = 0.024) and is
  indistinguishable from plain scatter search on everything. It removes the cost and keeps
  nothing.
* A deferral of 3 has the most successes, 24 against 17 for plain scatter search (13 seeds
  it alone won against 6), and 16 of 20 on Shahrezaei_PNAS2008, the noisiest problem, where
  its final error beats plain scatter search in 15 of 20 seeds (p = 0.041). Its medians on
  two problems are worse, not significantly. Nothing separates it from 2 at this sample
  size (error 43 against 36; successes 12 against 6).
* Five is the worst of the three on final error and the only setting significantly worse
  than another.

Three is chosen over two because it is the setting at which the feature shows the benefit it
was built for rather than merely costing nothing, and because the evidence against it is not
significant. It is a default, not a limit: a conf that sets `ss_noise_max_draws` is
unchanged, and the runner scores any other value in half an hour
(`run_baseline.py run --methods ss_noise --set ss_noise_max_draws=N --as ss_noise_dN`).

## Consequences

* `ScatterSearchConfig.ss_noise_max_draws` defaults to 3; the runtime fallback, the config
  documentation, the changelog and the effective-config golden follow. No other behaviour
  changes; a deterministic fit is untouched.
* The benchmark's baseline rows for `ss_noise` were measured at the old default and stay as
  they are, labelled by what ran; the deferral-3 rows in `results/ss_noise_20seeds.json`
  are the measurement of the new default on the four problems.
* Left open, on the benchmark's evidence: accepting a child that leads on the mean while the
  draws continue, and re-drawing only for the contests that matter most. Either could recover
  more of the cost; neither is needed to justify this change.

## Alternatives considered

* **Keep five.** It is the one setting the benchmark says is worse than another.
* **Two.** Equal to three on every test at this sample size and better on the pooled median
  error; but at two the feature is indistinguishable from not having it, which is a reason
  to turn it off, not to keep it on by default.
* **Turn the feature off by default.** Its cost was the deferral, not the machinery; with the
  deferral at three the machinery is the only setting that beat plain scatter search on
  successes. Off remains one key away (`ss_noise_handling = 0`).
