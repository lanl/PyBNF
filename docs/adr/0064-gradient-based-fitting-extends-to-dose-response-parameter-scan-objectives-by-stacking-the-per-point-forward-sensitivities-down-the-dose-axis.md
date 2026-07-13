# Gradient-based fitting extends to `parameter_scan` (dose-response) objectives by stacking the per-point forward sensitivities the scan already computes down the dose axis (issue #476)

**Status: Accepted (2026-07-12).** Extends the #385 gradient epic (layers A–J:
`pybnf/gradient/`), whose forward-sensitivity plumbing (#447 tensor →
`Data.output_sensitivities`, #448 routing, #449 assembly) covered **time-course**
objectives only. A dose-response objective — a `parameter_scan` that sweeps one
model parameter and scores the per-dose readout — could not be a gradient target:
the scan `Data` was assembled without a sensitivity tensor, so gradient assembly
raised `GradientNotSupported` (`assembly.py` → `gradient_base.py`) and the whole
fit aborted. For the **default** dose-response path this was a wiring gap, not a
math gap.

## The tensor was already computed, then discarded

The default synthesized dose-response experiment (`config.py`) uses
`steady_state=1` with `ss_method` omitted → the **parity / integrate-to-steady-state**
strategy (`_scan_parity_steady_state`), which runs each dose point via
`Simulator.run(steady_state=True)` on a **sensitivity-configured** simulator
(`_make_scan_simulator` threads `sensitivity_params`/`sensitivity_ic` for every
ODE run on the gradient path, exactly as the time course does). So the
`∂obs(dose_i)/∂θ` tensor was **already computed at every dose point** — and then
thrown away at row assembly: `_scan_result_to_row` read only the observable /
expression values and built a bare `Data` with `output_sensitivities = None`,
whereas the time-course `_result_to_data` attaches the tensor.

Because PyBNF hand-rolls the dose-response as a per-point `run()` loop (not
`Simulator.parameter_scan()`), bngsim's own `parameter_scan` sensitivity refusal
is never reached here — each point is an ordinary sensitivity-bearing `run()`.

## The decision

### Stack the per-point final-row sensitivities down the dose axis

A dose-response `Data` has one row per swept dose, each row being the **final**
integrated state (the equilibrium, or end-of-run) of an independent,
reset-to-seed per-point run. A new `pybnf.data.stack_scan_sensitivities` takes
each point's per-point `OutputSensitivities` (the full per-point tensor,
extracted through the unchanged `_extract_output_sensitivities`), reads its
**last integrated row**, and stacks those rows down a new leading axis — yielding
a `(n_doses, n_selectors, n_axis)` tensor, the **same layout** a time-course
`Data` uses, with the swept dose occupying the row slot. The scan strategy
collects the per-point tensors into a transient `_pending_scan_sens`, and
`_run_parameter_scan` stacks them onto the assembled scan `Data`.

The swept dose is the data's **independent variable**, not a fitted θ, so
`∂obs/∂θ` per dose is well-posed, and gradient assembly consumes the tensor
**unchanged**: it already addresses the row axis by the sim row that matches each
experimental point's independent variable (`_sim_row_for`), so a dose row is
indexed exactly as a time row. No assembly change; the normalization chain rule
(ADR-0053), per-condition routing factor (#448), and native→sampling transform
(ADR-0029) all thread through as-is, now over dose rows.

### Scope — reset-to-seed strategies only; refuse the rest cleanly

Only strategies where each dose point is an **independent, reset-to-seed** ODE
run have a well-posed, self-contained per-point sensitivity:

* **parity / integrate-to-steady-state** (`steady_state=>1`, `ss_method` omitted —
  the default) and **independent fixed-time** (`_scan_independent`, sequential
  branch) — **supported**.

The rest **refuse cleanly on the gradient path** — a PyBNF-level error naming the
alternative, gated by the #475 scored-action test (`_action_bears_sensitivities`)
so an *incidental* (unscored) scan of the same shape still runs sensitivity-free:

* **Newton / KINSOL** (`ss_method=>"newton"`): the algebraic steady-state solve
  performs no forward-sensitivity integration. Refuses before building any
  sensitivity-configured simulator, pointing at the differentiable parity default.
* **continuation / bifurcate** (`reset_conc=>0`): each point's initial state is
  the previous point's θ-dependent end state, so a correct sensitivity seed would
  have to be chained point-to-point (`dx0/dθ ≠ 0`) — not yet supported.
* **protocol** (`method=>"protocol"`): per-point multi-phase protocol
  sensitivities are not yet wired for a scan.
* **carried-state** (a pre-equilibration `simulate` advanced the model off seed;
  ADR-0062): already refuses (bngsim rejects sensitivity-configured carried-state
  scans).

### `run_batch` bypass and the SBML/Antimony twin

An ODE scan on the gradient path takes the **sequential** per-point `run()` loop:
`run_batch()` cannot return forward output sensitivities, so a scored scan that
would otherwise batch (`_scan_independent`, ≥4 points) is forced sequential —
which both collects the tensor and avoids a doomed `run_batch()`+fallback. The
SBML/Antimony backend (`bngsim_sbml_model`) gets the identical stacking on its
`species:` selectors; its `ParamScan` already runs each dose as an independent
reset-to-seed ODE `run`.

## Scope

* **In:** parity-steady-state and independent reset-to-seed dose-response
  gradients on both the net and SBML/Antimony backends; parameter and
  initial-condition sensitivity axes; the per-condition factor, cross-experiment
  sum, normalization chain rule, and native→sampling transform (all inherited
  from the time-course path, validated by FD).
* **Out (refused with a message):** Newton/KINSOL, continuation/bifurcate,
  protocol, and carried-state scans on the gradient path. Newton would need
  implicit-function-theorem sensitivities from the fixed-point Jacobian
  (`dx*/dθ = −(∂f/∂x)⁻¹ ∂f/∂θ`); continuation would need sensitivity
  seed-chaining across points.

## Consequences

* **The steady-state sensitivity convergence caveat.** The CVODE
  integrate-to-steady-state early-stop triggers on the *state* residual
  `‖f(t,y)‖₂/n`, not the *sensitivity* residual. In a stiff system `dx_ss/dθ` can
  converge on a slower timescale than `x_ss`, so the recovered tensor at the
  truncation row could be under-converged and yield subtly wrong gradients. The
  acceptance gate validates the recovered tensor against a central finite
  difference of PyBNF's own loss on a birth-death dose-response (analytic oracle
  `S* = dose/k_deg`, `∂S*/∂k_deg = −dose/k_deg²`); a user hitting a stiff regime
  should tighten the run or fall back to a gradient-free fit.
* **Purely additive.** The scalar (metaheuristic) path never activates the
  request, so a scalar-path scan `Data` is byte-identical — `stack_scan_sensitivities`
  returns `None` when every point is scalar-path (or any point lacks a tensor),
  leaving `output_sensitivities` `None`.
* Composes with #475: an incidental (unscored) scan of any shape runs
  sensitivity-free and never aborts a gradient fit.

## Validation

`test_bngsim_output_sensitivities.py` (analytic dose-axis oracle on the
birth-death net, parity + independent, scalar-path byte-identity, Newton /
continuation refusals), `test_gradient_assembly.py` (FD acceptance gate: assembled
dose-response objective gradient vs. central differences of PyBNF's own loss,
linear + log10 scale, two experiments), `test_bngsim_sbml_bridge.py` (SBML scan
dose-axis tensor vs. FD, scalar-path byte-identity), and `test_data_class.py`
(pure-numpy `stack_scan_sensitivities`, no bngsim).
