# Lesson 6 — When a gradient fit is refused, and how to fix it

**Feature:** the gradient-refusal path; making a model gradient-friendly · **Backend:** bngsim · **Difficulty:** ★★☆

A first-order system driven by an input that rises at time `tau`:

```
dX/dt = J(t) − k·X ,   J steps from J_base to J_base + J_step around t = tau
```

We fit it **three ways** — and the contrast is the lesson.

## Files

| File | What it is |
| --- | --- |
| [`step_input.bngl`](step_input.bngl) | the **hard-step** model; input is `if(t < tau, …)`. |
| [`step_input.exp`](step_input.exp) · [`step_input_smooth.exp`](step_input_smooth.exp) | data for each model. |
| [`step_input_de.conf`](step_input_de.conf) | hard step, **differential evolution** — works. |
| [`step_input_trf_refused.conf`](step_input_trf_refused.conf) | hard step, **`trf`** — refused. |
| [`step_input_smooth.bngl`](step_input_smooth.bngl) | the **smooth-step** model; input is a logistic sigmoid. |
| [`step_input_smooth_trf.conf`](step_input_smooth_trf.conf) | smooth step, **`trf`** — works, and recovers `tau` too. |

## Run them

```bash
pybnf -c step_input_de.conf              # hard step + de   -> recovers k, J_base, J_step
pybnf -c step_input_trf_refused.conf     # hard step + trf  -> errors out (the lesson)
pybnf -c step_input_smooth_trf.conf      # smooth step + trf -> recovers k, J_base, J_step, tau
```

## The three-way contrast

**1. Metaheuristics don't care about smoothness.** Differential evolution never
asks for a gradient, so the hard `if()` input is no obstacle — `step_input_de.conf`
recovers the parameters cleanly.

**2. A gradient optimizer is refused on the hard step, for a specific reason.**
`trf`/`lbfgs` consume bngsim's forward parameter sensitivities, and bngsim's
sensitivity engine does not support the `if()` construct — it raises
*"expression 'Input_X' … uses unsupported construct: if() conditional."* PyBNF
surfaces that as an actionable refusal.

**3. Smooth the step, and the gradient path opens up.** Replace the hard `if()`
with a differentiable approximation — a logistic sigmoid,
`J(t) = J_base + J_step · σ(s·(t − tau))`, `σ(x) = 1/(1+e^{−x})`. Now bngsim can
differentiate it, so `trf` fits happily. **Bonus:** because the transition is
differentiable, the gradient can recover the transition time `tau` itself — which
the hard-`if()` model could never expose to a gradient.

### The catch (it's a modeling choice, not a free lunch)

The sigmoid has a **sharpness** `s_sharp`. Turn it up and you approach a true
step — but the ODE stiffens and the sensitivities near `tau` steepen, so you trade
a *refusal* for *numerical* trouble. Turn it down and it fits easily but looks less
like a step. Choose it to be smooth *enough* to differentiate, not so sharp you
recreate the cliff. And because data rarely constrains how sharp the switch is,
`s_sharp` is usually **fixed** (a modeling assumption), while the physical
parameters — and now `tau` — are fitted. (Other smooth steps work too: `tanh`,
`arctan`, a Gaussian CDF/`erf`, a Hermite smoothstep.)

## Takeaway

Reach for `trf`/`lbfgs` on smooth models (Lessons 1, 3). On a non-smooth one you
have two good options: use a gradient-free algorithm (`de`/`pso`/`ss`/…), **or**
replace the non-smooth construct with a differentiable approximation and keep the
gradient path — often the better choice, since it can also fit the transition
itself. (First-class `if()` sensitivity support is tracked upstream in
PyBNF-Private #250; if it lands, the hard-step model becomes gradient-fittable too.)
