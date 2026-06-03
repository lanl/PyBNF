# The mandatory method contract is `Algorithm`'s two abstract methods; the run loop is shared, not pluggable

Status: Accepted (implemented in M2.2 move 1).

A PyBNF fit type ("method") plugs into the framework by **subclassing
`Algorithm`** and implementing exactly two methods — `start_run() -> list[PSet]`
and `got_result(Result) -> list[PSet] | 'STOP'` — plus registering itself with a
config schema (`@register_fit_type(..., schema=...)`; ADR-0002, ADR-0005).
Everything else the framework needs (`reset`, `add_iterations`, `cleanup`,
`get_backup_every`) has an overridable default. The **run loop** — job
submission, `as_completed` draining, backup cadence, best-fit save, sim-dir
teardown — lives in `Algorithm.run()` and is **shared by every method, not
replaceable**. We codify the two methods as `@abstractmethod` (they were
`NotImplementedError` stubs) and document the contract on the `Algorithm` class.

## Considered Options

- **A standalone `Protocol`/ABC decoupled from `Algorithm`** (composition-style),
  with `Algorithm` as merely one provider of it. Rejected: it only earns its keep
  if a method may bring its **own** run loop — a capability we do not want to
  promise or test. Every method on the horizon (nested sampling, SMC, HMC) wants
  different `start_run`/`got_result` bodies, not a different outer loop. A
  decoupled Protocol would invite run-loop forks and divergent backup/teardown.
- **Leave the `NotImplementedError` stubs.** Rejected: they do not fail at
  construction, do not document the contract, and do not signal "this is THE
  extension surface."

## Consequences

- "Add a method = subclass `Algorithm`, implement two methods, register with a
  schema." The (prospective) Sampler Toolkit (ADR-0009) is orthogonal — composed
  helpers a method MAY use, never part of the contract.
- `got_result`'s `list | 'STOP'` stringly-typed sentinel is kept as-is
  (behavior-preserving). Replacing it with a typed signal touches all four
  `got_result` returns + the run loop and is a separate, later cleanup.
- The change is a no-op for existing leaves (each already implements both
  methods); the full suite is the guard.
