# No Sampler Toolkit yet — stepping logic stays per-method

Status: Accepted (M2.2).

The M2 plan anticipated harvesting a "Sampler Toolkit" of reusable stepping
blocks (Metropolis kernel, proposals, tempering, cooling) and refactoring the
samplers onto it. On inspection the harvest does not clear its own
**≥2-real-users** bar, so we do **not** build it in M2.2 — not even a minimal
version.

## Considered Options

- **Harvest a minimal toolkit** (a gaussian random-walk proposal + a Metropolis
  accept) and route `mh`/`pt` + the rewritten `sa` through it. Rejected: the
  shared surface is ~15 lines of *textbook* math (not project-specific knowledge
  that must be kept in sync), and the realistic user set is **closed and
  shrinking** — only `mh`/`pt` and the deprecated `sa` fit these two functions.
  `am` (adaptive multivariate-normal proposal, differently-written accept) and
  `dream` (DE-archive donors + snooker Hastings correction) carry their own
  kernels and never will; new samplers will likewise bring their own (as `am` and
  `dream` already did). The seam costs a module, an indirection, and a refactor of
  working samplers, for near-zero dedup with no growth path.
- **Full consolidation** (one parameterized kernel for all four, refactor
  `am`/`dream` too, rebaseline numerics off a posterior-invariance KS check
  instead of byte-identity). Rejected: large, risky, and abandons the
  byte-identical discipline for a cosmetic win.

## Consequences

- Each sampler owns its propose/accept **by design**; that is intentional, not
  debt.
- CONTEXT.md's "Sampler Toolkit" and "Metropolis Kernel" terms are reframed as
  **prospective, not current** — they name a concept to harvest *if* a future
  sampler genuinely wants shared stepping, to its real shape then.
- Genuinely-shared, substantial, oracle-testable code IS still extracted —
  convergence diagnostics (R-hat / ESS) move to a top-level `pybnf/diagnostics.py`
  (M2.2 move 5). The bar is **substantial + already-shared + project-specific +
  oracle-testable**, which diagnostics clear and the kernel does not.
