# Scored Newton/KINSOL steady-state dose-response scans become differentiable by consuming bngsim's observable-level steady-state sensitivities (issue #478)

**Status: Accepted (2026-07-14).** Extends ADR-0064 (#476), which stacked the
per-point forward sensitivities of a **reset-to-seed** dose-response scan down the
dose axis but deliberately **refused** the Newton/KINSOL accelerator
(`ss_method=>"newton"`/`"kinsol"`) on the gradient path: a KINSOL *algebraic*
steady-state solve `f(x)=0` performs no forward-sensitivity **integration**, so —
unlike the parity integrate-to-steady-state default — it produced no
`∂obs(dose)/∂θ` trajectory for gradient assembly to consume. This ADR makes a
scored Newton scan differentiable too, so `ss_method=>"newton"` is a genuine speed
win under a gradient fit (`fit_type = trf`/`lbfgs`) rather than a
fall-back-to-parity refusal.

## The hard part was already computed backend-side

The steady-state parameter sensitivity `dY_ss/dp` is exactly the
implicit-function-theorem derivative `dx*/dθ = −(∂f/∂x)⁻¹ ∂f/∂θ`, and bngsim's
KINSOL solve **already returns it from the analytical Jacobian** — *exact*, not a
finite difference. What ADR-0064 lacked was the last mapping: `SteadyStateResult`
exposed that sensitivity only at the **species** level (`dY_ss/dp`, shape
`(n_species, n_params)`), with no observable/expression accessor, whereas the
gradient path needs `∂g/∂θ = (∂g/∂x)·(dx_ss/dθ)` keyed by
`observable:`/`expression:` selectors — the species sensitivity mapped through the
observable (stoichiometric group) / function Jacobian `∂g/∂x`.

## The decision fork — where `∂g/∂x` is applied

Issue #478 framed the choice as:

* **Option A — PyBNF-only.** Map species→observable in PyBNF: linear group
  coefficients for `begin groups`, the function chain rule for `begin functions`.
  Self-contained, but PyBNF must own the observable/function Jacobian — trivial for
  linear groups, decidedly not for arbitrary `begin functions` expressions.
* **Option B — small bngsim enhancement.** Have `SteadyStateResult` expose an
  `output_sensitivities(selectors, axis=…)` accessor mirroring the CVODE `Result`.
  bngsim already computes the observable/function map on every run, so the PyBNF
  side reduces to reading the tensor exactly as the parity path does.

### Chosen: Option B — consume bngsim's observable-level accessor

bngsim ≥ 0.11.35 (companion lanl/bngsim#12) ships
`SteadyStateResult.output_sensitivities(selectors, axis="parameter")` plus
`observable_names`/`expression_names`, projecting `dY_ss/dp` through the **exact**
linear group map for observables and a finite-difference total derivative for
`begin functions` globals, validated backend-side against a CVODES
forward-sensitivity run at steady state. PyBNF consumes it rather than
re-implementing `∂g/∂x` — the function-expression Jacobian in particular is not
something a fitting front-end should re-derive.

## The PyBNF wiring

A new `BngsimModel._extract_ss_output_sensitivities` mirrors the CVODE
`_extract_output_sensitivities` — same `observable:`/`expression:` selectors, same
`OutputSensitivities` shape — differing only structurally:

* **Singleton time axis.** A `SteadyStateResult` is a single equilibrium point, so
  its tensor has no leading time axis (`(n_selectors, n_params)`). We prepend a
  singleton row → `(1, n_selectors, n_params)`, the same rank a time-course tensor
  has, so `stack_scan_sensitivities` picks the final row (`[-1]`) identically for a
  converged Newton point and a CVODE fallback point.
* **IC axis is structurally zero.** A stable steady state forgets its initial
  conditions (`∂x*/∂x(0)=0`, ADR-0064/#457), so an IC-seed fit parameter
  contributes no gradient at equilibrium. bngsim declines the `ic` axis on a
  `SteadyStateResult` for exactly this reason; PyBNF reports it as an explicit zero
  tensor when IC sensitivities were requested (parity with the integrate path,
  whose IC sensitivities have decayed to ~0), and `None` otherwise.

`_scan_newton_steady_state` drops the ADR-0064 refusal and, on the scored gradient
path (`_action_bears_sensitivities()`):

* calls `steady_state(sensitivity_params=req.params)` per dose;
* on KINSOL convergence, reads the slice from the `SteadyStateResult` accessor
  (**not** the tiny post-solve eval `run(t_span=(0,1e-10))`, whose fresh-IC
  sensitivities are ~0);
* on the **non-convergence fallback** (a long CVODE time-course to `t_end`, which
  ADR-0064's parity path already makes differentiable) reads the slice via the
  unchanged `_scan_point_sensitivities`/`_extract_output_sensitivities` — the two
  extractors are byte-shape-identical, so a scan whose points *mix* convergence and
  fallback stacks cleanly down one dose axis;
* collects the per-point slices into `_pending_scan_sens`, which `_run_parameter_scan`
  stacks onto the scan `Data` via `stack_scan_sensitivities` — **no gradient-assembly
  change**.

### Sequential on the gradient path

The threaded batch path (`_run_ss_scan_threaded`, ≥4 points) submits
`steady_state()` to a `ThreadPoolExecutor`. The KINSOL *sensitivity* solve is not
confirmed thread-safe, so a scored scan is forced onto the sequential per-point
loop — exactly as ADR-0064 forces the independent scan off `run_batch` (which
cannot return sensitivities) on the gradient path. The scalar (metaheuristic)
Newton scan keeps the threaded path unchanged.

### Capability gate, not a version wall

`ss_method=>"newton"` runs on any supported bngsim in a scalar fit; only the
**gradient** path needs the new accessor. A centralized probe
`BNGSIM_HAS_SS_OUTPUT_SENS` (`_bngsim_caps` → `_runtime`, a `hasattr` on
`SteadyStateResult.output_sensitivities` — `capabilities()` has no dedicated key
for it) gates the gradient Newton scan: a build without it refuses cleanly with an
upgrade hint (`bngsim>=0.11.35`) rather than an `AttributeError` deep in the
backend. The `pyproject` floor is bumped to `bngsim>=0.11.35`, but the gate keeps a
stale-install refusal actionable.

## Scope

* **In:** scored `ss_method=>"newton"`/`"kinsol"` reset-to-seed dose-response
  gradients on the net backend; parameter axis from the exact KINSOL `dY_ss/dp`
  mapped through `∂g/∂x` (linear groups + function expressions); the IC axis as a
  structural zero; the per-condition factor, cross-experiment sum, normalization
  chain rule, and native→sampling transform (all inherited unchanged from the
  time-course / parity path). The KINSOL→parity non-convergence fallback is
  differentiable and consistent with the converged path.
* **Out (still refused with a message, per ADR-0064):** continuation/bifurcate
  (`reset_conc=>0`, needs per-point sensitivity seed-chaining `dx0/dθ ≠ 0` — its
  own follow-up), `method=>"protocol"`, and carried-state (pre-equilibration,
  ADR-0062) scans on the gradient path.

## Consequences

* **No steady-state convergence caveat on the converged path.** Unlike the parity
  integrate-to-steady-state path (ADR-0064: the CVODE early-stop triggers on the
  *state* residual, so `dx_ss/dθ` can be under-converged in a stiff system), the
  KINSOL solve returns the *exact* implicit-function-theorem sensitivity from the
  fixed-point Jacobian — no sensitivity-convergence lag. Only the (rare)
  non-convergence CVODE fallback inherits ADR-0064's caveat.
* **Purely additive.** The scalar path never activates the request, so a
  scalar-path Newton scan `Data` is byte-identical; `stack_scan_sensitivities`
  returns `None` when any point lacks a tensor.
* Composes with #475: an *incidental* (unscored) Newton scan of any shape runs
  sensitivity-free and never aborts a gradient fit.
* **Partly supersedes ADR-0064**, whose "Out" list refused Newton/KINSOL.

## Validation

`test_bngsim_output_sensitivities.py` — analytic dose-axis oracle on the
birth-death net (`∂S*/∂k_deg = −dose/k_deg²`) for `ss_method=>"newton"`, a
**multi-species** observable oracle on a new two-species cascade fixture
(`Ptot = 2A + 3B`, so the value `5·dose/k_deg` and sensitivity `−5·dose/k_deg²`
both pin the `∂g/∂x = [2, 3]` map a bug would shift), scalar-path byte-identity,
and the capability-gate refusal (patched `BNGSIM_HAS_SS_OUTPUT_SENS = False`).
`test_gradient_assembly.py` — an FD acceptance gate on the **multi-species**
Newton dose-response (assembled objective gradient vs. central differences of
PyBNF's own loss, linear + log10 scale, two experiments), which the
identity-observable birth-death net of the #476 gate could not catch a `∂g/∂x`
bug with.
