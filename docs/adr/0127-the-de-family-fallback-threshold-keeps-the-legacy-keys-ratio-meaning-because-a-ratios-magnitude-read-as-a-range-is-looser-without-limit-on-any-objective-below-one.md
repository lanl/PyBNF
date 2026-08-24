# The Differential Evolution family's fallback threshold keeps the legacy key's ratio meaning, because a ratio's magnitude read as a range is looser without limit on any objective below one (issue #648)

## Status

Accepted. Corrects one claim in ADR-0115, and changes nothing else it decided.

## The defect

ADR-0115 replaced the family's convergence test with an absolute objective range and gave
it a new key, `de_tolfun`. Both were right. It also decided that an unset `de_tolfun`
falls back to `stop_tolerance`, so an existing configuration keeps the number it had, and
justified reading that number as a range like this:

> A fit that wants the old *relative* semantics has no equivalent — but that semantics was
> only ever meaningful for a positive objective, where an absolute range at the same
> magnitude is a stricter, well-defined stop.

The claim that the range is stricter is true only above an objective of 1. Below it the
range is looser, and without limit, because the two thresholds diverge as the objective
falls:

| objective the population reaches | ratio 0.002 stops at a spread of | range 0.002 stops at a spread of | range is |
|---|---|---|---|
| 1000 | 2 | 0.002 | 1000x stricter |
| 1 | 0.002 | 0.002 | the same |
| 2e-05 | 4e-08 | 0.002 | 50000x looser |

A sum of squares fit on well scaled data lands in the bottom row as a matter of course.
There the population satisfies the stop almost as soon as it is scored, and the run ends
while its parameters are still far apart.

## Observed

Tutorial lesson 25 fits a transit compartment pharmacokinetic model, observing only the
central compartment. It reaches an objective of 2.19e-05 and stops there, reporting
`k_transit` as 11.18 against a true 12.76 and `k_abs` as 11.50 against a true 9.11. The two
rates trade against each other, so the fit that stops early fits the data well and reports
a wrong answer with nothing wrong on its face. `k_elim` is recovered to three digits, which
is what makes the failure hard to see.

The lesson passes at v1.7.0 and at cd6e8531, and fails at 8e006374, which is ADR-0115's
commit. Adding `de_tolfun = 1e-10` to the unchanged lesson recovers all three rates.

Of the 58 tutorial configurations that use `de` or `ade`, none sets either key, so all 58
ran on the reinterpreted number. 45 of them are checked for parameter recovery and 44 still
recovered, so the reach is one fit in 45. The reach is narrow because most fits separate
their parameters before the stop can cut them short. It is not narrow by design.

## The decision

**Whether `de_tolfun` was set decides what the threshold means.**

* An **explicit** `de_tolfun` is an absolute range in objective units, chosen by an author
  who can see their own objective's scale. It is honoured as written, at any sign and any
  scale. ADR-0115's key keeps exactly the meaning it was given.
* An **unset** `de_tolfun` falls back to `stop_tolerance`, and that key keeps the meaning it
  has always had. Where the objective is positive it is a dimensionless ratio, applied as
  `max - min <= tol * min`. That is the `max / min <= 1 + tol` this family used before #561,
  written without the division so an all-zero population cannot divide by zero. Where the
  objective is not positive a ratio means nothing, so the number is read as an absolute
  range, which is the branch ADR-0115 built and the one #561 needs.

The test is still assessed over the finite fitnesses only, and the island guard still holds.
Both are ADR-0115's, both were correct, and neither changes here.

## Why not a different default

Deriving a range from the objective the population has reached is the same arithmetic as
the ratio, reached by a longer route and with a new number to explain. Choosing a smaller
absolute default only moves the problem: any fixed range is wrong by the ratio of two
objectives' scales, and PyBNF fits objectives spanning many decades.

Keeping the ratio was rejected wholesale by ADR-0115 because it is the source of #561. It
is not. #561 is a ratio applied where a ratio has no meaning. Applied where it does, a ratio
is the only one of the two forms that gives one value the same meaning on every fit, which
is what a default has to do.

## Consequences

* Every fit that does not set `de_tolfun` and whose objective is positive converges exactly
  as it did at v1.7.0, with #561's fix to failed simulations retained.
* Every likelihood fit reaching #561's branch is unaffected. All six of ADR-0115's
  regression tests use non-positive fitnesses and pass unchanged.
* A configuration that set `de_tolfun` explicitly is unaffected.
* A configuration that set `stop_tolerance` explicitly gets that number read as a ratio
  again, which is what it meant when it was written.
