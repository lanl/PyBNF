# `sa` (Simulated Annealing) is an optimizer: minimize the raw objective, extracted to `optimizers/`

Status: Accepted (implemented in M2.2 moves 3–4).

Simulated Annealing was historically built on the Bayesian sampler base
(`BasicBayesMCMCAlgorithm`, registered with `kwargs={'sa': True}`) and evaluated
the **posterior** (prior + likelihood) in its Metropolis accept. That makes it,
for `normal`/`lognormal` priors, a silent **MAP estimator** — behavior no other
PyBNF optimizer (`de`, `pso`, `ss`, `sim`) applies; those minimize the raw
objective and use the prior only for initial sampling. For `uniform`/`loguniform`
priors the prior term is a constant that cancels in the acceptance ratio plus a
`-inf` outside the box that proposal reflection already prevents — i.e. a no-op.

We rewrite `sa` **to spec as a true optimizer**: minimize the raw objective, box
constraints via proposal reflection + bounds (unchanged), real temperature /
cooling, standalone config out of the MCMC family. It lands in
`optimizers/simulated_annealing.py` as `SimulatedAnnealing(Algorithm)` with a
clean 1:1 registry binding (no `kwargs`), `family='optimizer'`,
`deprecated=True`.

## Considered Options

- **Faithful byte-identical extraction** (relocate sa's code, preserve every RNG
  draw, keep the posterior accept). Rejected: byte-identical relocation is the
  right discipline for complex code moved semi-blind (the M1 god-file split), but
  `sa` is ~40 lines, fully understood, deprecated, and partly *dubious* — a
  faithful move would enshrine the MAP-via-prior confusion, the vestigial
  constraint caching (cached, never read for sa), and the `beta`-as-temperature
  naming. A clean rewrite-to-spec with a functional guard is the better tool.
- **Keep evaluating the prior density** (so sa stays a MAP optimizer). Rejected:
  inconsistent with every other optimizer; conflates "where to start sampling"
  with "what to optimize."
- **Delete `sa`** (it is deprecated). Not taken: removal is a separate later
  decision (see Cross-cutting in the plan); the rewrite makes `sa` a correct,
  consistent optimizer in the meantime.

## Consequences

- **Intended behavior change** for fits with `normal`/`lognormal` priors and
  `fit_type = sa`: results differ (objective minimization vs MAP). Recorded in
  CHANGELOG; the golden-config `sa` entry is regenerated knowingly.
  `uniform`/`loguniform`-prior fits are numerically unaffected (the prior term was
  already a no-op there).
- The guard shifts from byte-identical to **functional**: a seeded `sa` converges
  to the known optimum on an all-uniform-prior analytical case (where the
  prior-drop is a numerical no-op), plus the rewritten cooling unit tests.
- `MCMCFamilyConfig.postprocess` sheds its `fit_type == 'sa'` branches and
  `BasicMCMCConfig` drops `cooling`/`beta_max` (sa-only). This is **byte-identical
  for mh/pt/am/dream** — those branches never fired for them; the golden net
  proves it.
- The `BayesianAlgorithm` prior machinery (`ln_prior`/`load_priors`) stays put;
  the rewritten `sa` needs none of it, so nothing is pushed down to the universal
  base — a cleaner outcome than the relocation first considered.
