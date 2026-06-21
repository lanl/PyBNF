# PyBNF has one log base — log10; natural log is the explicit LN (toward #407 observables)

PyBNF uses **log10** everywhere a parameter is log-scaled: `logvar`,
`loguniform_var`, `lognormal_var`, the proposal arithmetic, and the prior
sampling space all transform `u = log10(θ)` (`priors/scale.py`). The one
exception was the **Additive-Noise Scale** (ADR-0011): its `LOG` member used
*natural* log (`np.log`), and the `lognormal` objfunc was `Gaussian(LOG, MEDIAN)`
— natural-log lognormal, matching `scipy.stats.lognorm`.

That single exception is a **units trap**, not a cosmetic quirk. The `lognormal`
objfunc reads its σ from the data's `_SD` column (or a free parameter). A user
steeped in PyBNF's log10 convention supplies a log10-scale σ; the natural-log
objfunc silently reads it as a natural-log σ — wrong by `ln10² ≈ 5.3`, with no
error, just a quietly wrong fit. PEtab v2 makes the gap concrete from the other
side: `observableTransformation` is `lin` / `log` (natural) / `log10`, so the
#407 observables adapter needs *both* natural-log and log10 noise scales anyway.
Documenting "these two `log`s differ" would be documentation-and-prayer; we remove
the ambiguity instead.

- **One log base in the core: log10. Natural log exists only as the explicit
  `LN`.** The noise scale gains three explicitly-named members — `LINEAR`,
  `LOG10` (`forward = np.log10`, `mean_offset = σ²·ln10/2`), and `LN`
  (`forward = np.log`, `mean_offset = σ²/2`). The ambiguous bare `LOG` symbol is
  **deleted**: every log scale now carries its base in its name. There is no
  context anywhere in PyBNF where a bare "log" silently means natural — so the
  wrong base cannot be assumed, because the ambiguous name no longer exists. This
  is the structural guarantee that replaces a documentation note.

- **The `lognormal` objfunc becomes `Gaussian(LOG10, MEDIAN)`.** It is now
  consistent with `lognormal_var` and every other PyBNF "log": a user's σ is a
  log10-scale standard deviation *everywhere*. This changes the objfunc's
  numerical values (natural → log10), a deliberate fix of the ~10-day-old M2.4
  objfunc; blast radius verified ~nil (zero config files in tests / examples /
  benchmarks set `objfunc = lognormal`; only its own M2.4 unit test pins the base,
  and it is updated). The natural-log lognormal density remains exactly testable
  via `Gaussian(LN, MEDIAN)` against the `scipy.stats.lognorm` oracle.

- **The Parameter Scale and the Additive-Noise Scale stay separate code, now
  sharing one base convention.** ADR-0011/CONTEXT.md keep them as distinct domain
  concepts (the space a *parameter is sampled* in vs the space a *measurement's
  noise* lives on) with separate classes; this ADR does **not** merge them. It only
  removes the base divergence — both honor "log = log10", and natural log is
  named `LN` on the noise side just as it would be if it ever appeared on the
  parameter side.

- **The PEtab adapter translates PEtab's vocabulary at the seam.** PEtab's `log`
  is natural and `log10` is log10, the opposite default from PyBNF. The #407
  observables adapter maps `lin → LINEAR`, `log → LN`, `log10 → LOG10` — the same
  kind of vocabulary translation `petab/parameters.py` already does converting
  PEtab's natural-log priors to PyBNF's log10 (`σ → σ/ln10`). The mismatch is
  confined to the adapter; the PyBNF core sees one base.

## Amendment (2026-06-20, #417 / ADR-0043)

The new-era `parameter:` record adds a first-class **`Ln` parameter scale**, selectable as
`parameter_scale: ln` (alongside `linear` / `log10`). This **amends the "one base = log10"
*simplification*** above — natural log is no longer noise-side only — but **keeps this ADR's actual
rule intact**: every log scale names its base explicitly (`log10` vs `ln`), and a bare ambiguous
`log` is rejected, so the units trap this ADR closed stays closed. The motivation is symmetric: just
as silently reading a log10 σ as natural was wrong, *forcing* a user who wants ln to convert to log10
by hand is user-hostile. The `Scale` abstraction already centralized the transform, so the families /
proposal arithmetic / prior density compose for free; only a few base-10 hardcodes that bypassed
`_scale` were routed back through it (ADR-0043's implementation note).

## Considered Options

- **Keep `lognormal` natural-log and document the difference.** Rejected — this is
  the "documentation and prayer" the units trap defeats: the ambiguous `LOG`
  remains, and a log10-thinking user is still silently misread. Correctness should
  not depend on the user reading a glossary note.

- **Convert log10 → natural-log by rescaling σ at the PEtab seam (no `LOG10`
  scale).** Rejected as the core solution: it is exact for a fixed σ but
  reparameterizes an *estimated* noise parameter into natural-log units (its
  reported value and bounds drift from the log10 the user expects), and it leaves
  PyBNF's own bare `LOG` ambiguous. A first-class `LOG10` scale is exact for every
  σ-source and is what makes "log = log10" a real invariant.

- **Merge the Parameter Scale and Additive-Noise Scale into one class.** Rejected:
  ADR-0011 deliberately separated them as different domain concepts; merging would
  re-smear "where a parameter is sampled" and "where a measurement's noise lives".
  Sharing the *base convention* without merging the *classes* gets the consistency
  without the conflation.

Relevant ADRs: **0011** (the per-point NoiseModel and its three axes — this amends
the additive-noise-scale axis from one ambiguous `LOG` to explicit `LOG10`/`LN`),
**0019** (the PEtab importer's adapter-translates-vocabulary discipline this
extends to `observableTransformation`). Follow-up: the #407 observables chunk that
consumes `LOG10`/`LN`.
