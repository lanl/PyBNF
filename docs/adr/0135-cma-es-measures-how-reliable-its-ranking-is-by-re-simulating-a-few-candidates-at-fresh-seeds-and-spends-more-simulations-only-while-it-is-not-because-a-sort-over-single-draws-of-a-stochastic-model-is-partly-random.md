# CMA-ES measures how reliable its ranking is by re-simulating a few candidates at fresh seeds, and spends more simulations only while it is not, because a sort over single draws of a stochastic model is partly random (issue #661)

## Status

Accepted and implemented (2026-09-10). The measurement is a module of its own so that
scatter search's reference set can be made noise-aware from the same piece (#660 step 3).

## The defect

CMA-ES ranks its population each generation and reads only that ordering: the mean moves
towards the best `mu`, the covariance adapts along the directions they lie in, and the step
size grows or shrinks by how far the ordering says the search has moved. For a stochastic
model every objective value is one draw. When the draw-to-draw spread is comparable to the
real differences between candidates, the sort is partly random, so the distribution is
pulled in arbitrary directions and, worse, the step-size adaptation reads noise as stagnation
and shrinks a step that should not shrink. The search then converges on nothing in
particular, quickly.

`smoothing` is not a fix. It runs every simulation the same fixed number of times for the
whole fit, which multiplies the cost of everything, and it cannot tell a generation whose
ordering is obvious from one whose ordering is not.

## The decision

### Measure the ranking's reliability, and react only when it is low

The treatment is Hansen, Niederberger, Guzzella and Koumoutsakos (2009). Each generation,
after its candidates are scored, a random subset of them (a tenth, floored at three) is
simulated again at fresh seeds. Each re-evaluated candidate's old and new values are ranked
together, and the rank change is compared with how far a value at that rank would move
under pure noise, the lower quartile of the possible rank changes at its position. The
average excess over the subset is the generation's measurement; filtered over generations
it is the uncertainty level. A deterministic function measures negative, since its two
values are adjacent; independent draws measure positive.

Above zero, every candidate of the next generation is simulated more times (the count grows
by 1.5, up to `cmaes_noise_max_evals`) and ranked on the mean of its draws, and the step size
is multiplied by `1 + 2/(n + 10)` after the cumulative step-length adaptation, holding it up
against the shrinkage noise induces. At or below zero the count shrinks back towards
one, as in the reference implementation. A re-evaluated candidate is ranked on the mean of
its two measurements.

The floor of three re-evaluated candidates matters at PyBNF's population sizes, where a
tenth of twelve is one. With two, the pure-noise limit at every rank is zero, so the
statistic can never come out negative: a deterministic ranking measures exactly zero and
any noise at all ratchets the draw count up to its cap with nothing to bring it down. With
three the limit is one at every rank, a deterministic ranking measures negative, and a
single swap measures zero.

The subset is random rather than the best-ranked, on purpose: the best-ranked of a noisy
generation are over-represented by lucky draws, whose re-evaluation would move them further
than a fair sample and overstate the noise.

### Only where running a parameter set again would give a different answer

The handling turns on when `_replicates_would_differ()`, the same gate the best-fit
confirmation stage (#659) uses: at least one stochastic model, and not every one pinned to
a trajectory by an `_honorbngl` seed policy. A deterministic fit queues exactly the jobs it
queued before, with the same names and the same replicate index 0, and is byte-identical.
`cmaes_noise_handling = 0` turns it off for a stochastic fit.

### A returned parameter set can ask for a fresh draw

Under the default `stochastic_seed = auto` policy a simulation's seed is derived from the
parameter values and the replicate index. A parameter set the fit has already scored,
submitted again through the run loop, would therefore reproduce the same trajectory and
measure nothing. The confirmation stage sidestepped this by submitting its own jobs; an
optimizer inside the `start_run` / `got_result` contract cannot. So a `PSet` gains a
`replicate_offset` an algorithm may set on a copy it returns, and `make_job` reads it when
the run loop passes none. It is a class attribute defaulting to 0, so an older backup reads
as unchanged, and it is not part of a PSet's identity, as its name is not. CMA-ES queues the
`e`-th draw of a candidate at offset `e * smoothing`, and the re-evaluation phase at offsets
past those, so every draw is fresh and the smoothing groups under it never collide.

### What the trajectory records

Every draw lands in the trajectory as its own entry, as every scored simulation always has,
so `sorted_params` shows a candidate's best draw and the confirmation stage at the end of the
fit re-runs the top candidates to settle the reported answer. The search itself ranks on
means, which is what this ADR is about; the reported answer was #659's.

## Cost

Per generation, a tenth of the population simulated once more while the ranking is
reliable, and up to `cmaes_noise_max_evals` simulations per candidate while it is not. On a
deterministic model, nothing.

## Consequences

* New `pybnf.algorithms.noise_handling.RankChangeNoise`: the measurement and the
  adaptation, picklable, with no knowledge of any optimizer.
* `CMAESAlgorithm` gains a re-evaluation phase between scoring a generation and updating
  the distribution, a per-candidate draw count, and the step-size factor; `CMAESConfig`
  gains `cmaes_noise_handling` (default 1) and `cmaes_noise_max_evals` (default 10),
  registered in the parse layer, the docs and the effective-config golden.
* `PSet.replicate_offset` and the `make_job` hook that reads it.
* Scatter search (#660 step 3) is the intended second consumer: its reference set needs an
  estimate and an uncertainty per member and a way to ask for more simulations where the
  ordering is in doubt, which is this measurement applied to a reference set instead of a
  generation.

## Verification

`tests/test_cmaes_noise_handling.py`: the measurement on hand-built values (a deterministic
function negative, independent draws positive, a small shift under large gaps negative), the
pure-noise limit by hand, the adaptation's growth, cap, shrinkage and filter; the seam
(`make_job` honours the offset, an explicit offset wins, smoothing shifts every replicate,
the offset is outside the identity); CMA-ES by hand with a scorer whose noise is keyed by
the replicate offset (a deterministic fit queues one draw per candidate and has no handling,
the switch turns it off, a stochastic fit re-evaluates a subset at fresh offsets and ranks
them on the mean, heavy noise raises the draw count and holds the step size, a quiet model
keeps one draw, candidates rank on the mean of their draws, a failed draw makes a candidate
infinite, smoothing spaces the offsets, the state pickles mid-generation, a restart clears
the phase); and the whole run loop with a noisy fake runner, which re-simulates at fresh
indices and still finds the mode.

## Prior art

Hansen, N., Niederberger, A. S. P., Guzzella, L. and Koumoutsakos, P. (2009). A method for
handling uncertainty in evolutionary optimization with an application to feedback control of
combustion. IEEE Transactions on Evolutionary Computation 13(1), 180 to 197. The reference
implementation is the `NoiseHandler` of Hansen's `cma` package, whose constants (a random
tenth re-evaluated, the lower quartile, a filter of 0.3, growth by 1.5 and shrinkage by its
fourth root, a step-size factor of `1 + 2/(n + 10)`) this follows.
