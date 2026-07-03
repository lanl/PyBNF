# Lesson 6 — When a gradient fit is refused: the step-input system

**Feature:** the gradient-refusal path; choosing gradient-free vs gradient · **Backend:** bngsim · **Difficulty:** ★★☆

A first-order system driven by an input that **steps up** at time `tau`:

```
dX/dt = J(t) − k·X ,   J(t) = J_base            for t < tau
                       J(t) = J_base + J_step   for t ≥ tau
```

The input is written with an `if(t < tau, …)` conditional. This lesson shows the
same model fit two ways — and one of them is deliberately refused.

## Files

| File | What it is |
| --- | --- |
| [`step_input.bngl`](step_input.bngl) | the model; the `if()` input is in `Input_X()`. |
| [`step_input.exp`](step_input.exp) | data: `Obs_X` at 25 times. |
| [`step_input_de.conf`](step_input_de.conf) | fit `k`, `J_base`, `J_step` with **differential evolution** — works. |
| [`step_input_trf_refused.conf`](step_input_trf_refused.conf) | the **same fit with `trf`** — refused. |

## Run them

```bash
pybnf -c step_input_de.conf              # recovers k, J_base, J_step
pybnf -c step_input_trf_refused.conf     # errors out — that's the lesson
```

## What to notice

- **Differential evolution doesn't care that the model is non-smooth.** It never
  asks for a gradient, so the `if()` input is no obstacle and it recovers the
  parameters cleanly.
- **The gradient optimizer is refused, for a specific reason.** `trf`/`lbfgs`
  consume bngsim's forward parameter sensitivities, and bngsim's sensitivity
  engine does not support the `if()` conditional construct — it raises
  *"expression 'Input_X' … uses unsupported construct: if() conditional."* PyBNF
  surfaces that as an actionable error pointing you at a gradient-free algorithm.
- **Takeaway:** reach for `trf`/`lbfgs` when the model is smooth (Lessons 1, 3);
  reach for `de`/`pso`/`ss`/… when it isn't (piecewise inputs, thresholds, dosing
  schedules). Adding `if()` support to the sensitivity engine is tracked upstream
  (PyBNF-Private #250); if it lands, this lesson flips to "gradient handles it too."
