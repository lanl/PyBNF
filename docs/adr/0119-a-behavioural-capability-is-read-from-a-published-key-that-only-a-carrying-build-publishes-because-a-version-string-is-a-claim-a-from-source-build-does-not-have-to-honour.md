# A behavioural capability is read from a published key that only a carrying build publishes, because a version string is a claim a from-source build does not have to honour (issue #558)

**Status: Accepted and implemented (2026-08-20).** Revises how one flag introduced by #536
is decided; the flag's meaning, and everything gated on it, are unchanged.

`pybnf/_bngsim_caps.py` is the one module allowed to ask what the installed bngsim can do.
It gated two capabilities two different ways, and the file itself explained why one of them
was unsafe — on the *other* flag:

| flag | how it decided |
|---|---|
| `BNGSIM_HAS_PER_SPECIES_ATOL` | a **name** probe: `hasattr(bngsim, 'AUTO')` |
| `BNGSIM_HAS_EVENT_SENS` | a **version floor** at exactly `(0, 12, 2)` |

> The probe is a name, not a version. The build that first carried #196 still declares
> 0.12.2 — the same string as the released wheel 25 commits behind it — so a version floor
> here would report *present* on an install that does not have it, and that is the expensive
> direction.

That is the whole argument, it is correct, and it applied verbatim to the flag next to it.

## The defect

bngsim bumps `__version__` at the **start** of a release cycle, not at the end. So the
version string identifies a *cycle*, not a build, and every from-source build made between
the bump and a given fix declares the same string as the release that carries it. For
`BNGSIM_HAS_EVENT_SENS` the fixes that set the line — lanl/bngsim#144 (an event assignment
reading the state lost its carried term `∂h/∂x·s⁻`) and lanl/bngsim#146 (a solver root
firing nothing rewound the state but not the sensitivity history) — landed inside the
0.12.1 → 0.12.2 window. A build from anywhere in that window before them declares `0.12.2`,
clears a `>= (0, 12, 2)` floor, and is reported as carrying them.

What that costs is asymmetric, and the asymmetry is the point. The flag does not gate a
*feature*; it gates **silent wrongness**. A build below the line does not refuse an event it
cannot differentiate — it returns a finite tensor with the event's contribution missing. So a
false *absent* is a refusal and a metaheuristic fit, while a false *present* is a gradient
fit that runs to completion, converges, and reports a number. The floor could only ever be
wrong in the second direction.

This is not hypothetical. Its sibling failure cost a real fit: on `Smith_BMCSystBiol2013`,
reaction 7 is `per_species_volume_scaling`, and without lanl/bngsim#160/#161 the whole model
declines the analytic `∂f/∂p`, all 25 columns fall back to CVODES difference quotients, and
every start times out to `inf`. The released 0.12.2 wheel lacks those commits and clears the
same floor. Thirteen hours produced nothing, and the capability layer could not tell the two
installs apart. The corpus note written up from that run reached the conclusion
independently: *probe capability, never version.*

## Why the obvious fixes are wrong

**Raise the floor.** To what? There is no released version that separates a build carrying
the fixes from a build declaring the same number and not carrying them — that is what "the
same number" means. Raising the floor to 0.13.0 would refuse the released 0.12.2, which is
correct, while still admitting a from-source 0.13.0 built before whatever the *next* fix is.
The instrument is wrong, not its setting.

**Probe a name.** This is what the neighbouring flags do, and it is unavailable here. What
separates these builds is behaviour, not API surface: `event_sensitivity_unsupported_reason`
has existed since bngsim's initial public release, and nothing in the Python namespace
appears or disappears at #144 or #146.

**Read a feature key.** `capabilities()` reports compiled backends and build options. It has
no key for "carries the event-sensitivity fixes", and it is not PyBNF's to add. This is the
honest reason the floor was written in the first place, and #558 states it as a defect
upstream of PyBNF rather than a PyBNF bug.

**Probe behaviour numerically.** Build a model with a state-reading event assignment, run it
with sensitivities, compare against a central difference — this is the measurement that
found the bug (`-10.96` where the model's own central difference says `-311.20`). It is the
only *direct* answer, and it costs a codegen and a compile at job start, needs the codegen
capability to be present at all in order to answer a question about a different capability,
and introduces a numerical tolerance that can itself be wrong. Rejected as the default; it
remains the right instrument for a capability with no witness at all.

## The decision

**Decide from a published capability key that only a carrying build publishes, and demote
the version to a veto.** `BNGSIM_HAS_EVENT_SENS` now resolves through three routes, first
match wins:

| # | route | when it answers |
|---|---|---|
| 1 | `features['event_sensitivities']` | if bngsim ever publishes a dedicated key — **both** directions, ahead of everything else |
| 2 | `features['effective_ic_sensitivity']` **and** version ≥ 0.12.2 | today |
| 3 | neither key published | absent |

Route 1 is issue #558's first ask and fires on no build that exists: naming the key costs
nothing and means the flag starts reading the real answer on the first bngsim that grows one,
without another PyBNF release. A published `False` there outranks route 2, or the key would
be decoration.

Route 2 is the load-bearing one, and it is a **witness**, not the capability.
`effective_ic_sensitivity` reports `Model.effective_ic_sensitivity` — the `∂x(0)/∂θ` reader
ADR-0100 consumes, which has nothing to do with events. It is usable as evidence because of
*where it landed*: lanl/bngsim#155 added it three commits after #146 and seven after #144,
inside the same release window, so a build that publishes it necessarily carries both. The
implication runs one way only:

```
publishes effective_ic_sensitivity  ⟹  carries #144 and #146
```

and the converse failure — a build from the handful of commits between #146 and #155, which
has the fixes and not the witness — reads as *absent* and is refused. That is the direction
the flag is allowed to be wrong in.

Route 3 is where the old floor used to live, and the change is that **the version can no
longer carry the answer on its own.** The witness shipped *in* 0.12.2, so a build claiming
0.12.2-or-newer that does not publish it is precisely the lying pre-release build. The floor
survives only as a conjunct on route 2, where it vetoes an incoherent build and preserves the
deliberate fail-closed reading of an unparseable version.

**No install that works today is refused.** Every released bngsim at or above the floor
publishes the witness; every one below fails both the witness and the floor, exactly as
before. The only installs whose answer changes are from-source builds inside the window —
which is the entire purpose.

**A refusal names the route, not the version.** `event_sens_probe()` returns the phrase that
answered, and `_require_differentiable_dynamics` prints it. Without that, a reader whose
bngsim already reports 0.12.2 is told to upgrade to 0.12.2.

### The trap one level down: a stale compiled core

An editable bngsim serves live Python from the source tree while loading `_bngsim_core*.so`
from a separately built artifact, with auto-rebuild deliberately off (lanl/bngsim#23). The
two halves drift, and during unrelated work an install reporting `0.12.2` was found with a
core binary three days older than the `.cpp` beside it. **Every check in this ADR passes
there** — the version, the feature keys, the witness — because nothing in the Python layer
moved. Only bngsim's own mtime comparison sees it.

bngsim already does that comparison and warns. The problem is *when*: PyBNF imports bngsim
while loading its own package, which is before `init_logging`, before the config is read, and
before the user has committed to anything, so the warning lands in import noise on a terminal
nobody is reading yet. `_report_bngsim_build()` repeats it at job start — the identity line
to the log unconditionally, the staleness report promoted to a console warning at verbosity 0
— where a reader can still stop a run that would otherwise spend hours producing statements
about code that is no longer in the tree. The identity line carries the build commit, which
is the only thing on hand that tells two installs declaring one version apart.

Reads of `bngsim._build_provenance` are guarded and memoized, and every failure is silent: it
is a private module, `gather()` walks the whole C++ source tree, and an install that cannot
answer must report no opinion rather than take a fit down.

## Consequences

* The general shape is reusable and is stated as the rule for this module: **a capability is
  decided by what the build says it can do; a version string is at most a veto.** The two
  neighbouring `hasattr` flags already followed it; this one now does too, and it does it
  through a published key rather than a name, which bngsim's "existing names will not be
  renamed or removed" contract covers.
* A witness is a proxy and its justification is a fact about *history*, not about semantics.
  It has to be written down at the probe, and it is. If bngsim publishes a dedicated key,
  route 1 retires the witness with no further change here.
* Fail-closed is now the behaviour on ambiguity in both places: an unpublished witness and an
  unparseable version both read absent. The cost of the wrong guess is a metaheuristic fit;
  the cost of the other wrong guess is a converged wrong answer.
* Not addressed here: surfacing bngsim's *analytic `∂f/∂p` decline* — the `Smith` failure —
  as a PyBNF-level warning. That is a different capability (lanl/bngsim#160/#161), it is
  reported by bngsim on the `bngsim` logger at codegen time rather than by `capabilities()`,
  and codegen happens on a worker at the first simulation rather than at setup, so it needs
  its own decision about where a worker-side decline reaches the user.
* The standing ask upstream is unchanged: behaviour-level feature keys in `capabilities()`,
  plus a resolved build identifier. Route 1 exists so that PyBNF consumes the first of those
  the day it appears; `bngsim_build_id()` consumes the second, which `_build_provenance`
  already computes.
