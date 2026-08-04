# Lesson 6 — A discontinuous input, and what the gradient path can and cannot take

**Feature:** gradient fitting through a discontinuity; what is still refused · **Backend:** bngsim · **Difficulty:** ★★☆

A first-order system driven by an input that rises at time `tau`:

```
dX/dt = J(t) − k·X ,   J steps from J_base to J_base + J_step at t = tau
```

We fit it **four ways** — and the contrast is the lesson.

## Files

| File | What it is |
| --- | --- |
| [`step_input.bngl`](step_input.bngl) | the **hard-step** model; input is `if(t < tau, …)`. |
| [`step_input.exp`](step_input.exp) · [`step_input_smooth.exp`](step_input_smooth.exp) | data for each model. |
| [`step_input_de.conf`](step_input_de.conf) | hard step, **differential evolution** — works. |
| [`step_input_trf.conf`](step_input_trf.conf) | hard step, **`trf`** — works, `tau` included. |
| [`step_input_ssa_refused.conf`](step_input_ssa_refused.conf) | hard step, **`trf`** on a **stochastic** experiment — refused. |
| [`step_input_smooth.bngl`](step_input_smooth.bngl) | the **smooth-step** model; input is a logistic sigmoid. |
| [`step_input_smooth_trf.conf`](step_input_smooth_trf.conf) | smooth step, **`trf`** — works. |

## Run them

```bash
pybnf -c step_input_de.conf              # hard step + de   -> recovers k, J_base, J_step
pybnf -c step_input_trf.conf             # hard step + trf  -> recovers those AND tau
pybnf -c step_input_ssa_refused.conf     # ssa + trf        -> errors out (the lesson)
pybnf -c step_input_smooth_trf.conf      # smooth step + trf -> recovers k, J_base, J_step, tau
```

## The contrast

**1. Metaheuristics don't care about smoothness.** Differential evolution never
asks for a gradient, so the hard `if()` input is no obstacle — and would not be
whatever the input looked like.

**2. Neither, it turns out, does the gradient path — for a discontinuity in
*time*.** A gradient optimizer needs `∂(prediction)/∂θ`, and a jump does not
destroy that. Two cases, both handled:

* **the jump is at a known time.** The crossing does not move when `θ` moves, so
  the forward sensitivity simply carries through it. Nothing special is required.
* **the jump is at a fitted time.** Now the crossing itself moves with `θ`, and
  the sensitivity picks up a term at the moment the switch fires. bngsim supplies
  it, which is why `step_input_trf.conf` estimates `tau` — from a hard step, with
  no smoothing at all.

**3. What *is* refused: a stochastic experiment.** `step_input_ssa_refused.conf`
scores an experiment with `method: ssa`. An SSA trajectory is a random walk; there
is no derivative of it with respect to the rate parameters to carry, so forward
sensitivities exist only for the ODE backend and PyBNF refuses up front. This is a
property of stochastic simulation, not a gap someone will close later — and the
remedy is the one this lesson has always taught: score it with a gradient-free
`job_type`.

**4. Smoothing is still a real option — as a *conditioning* choice.** The sigmoid
variant fits the same parameters to the same accuracy. What differs is what each
costs the integrator: a true discontinuity makes the solver stop, restart, and
take small steps around the crossing, and a sigmoid trades that for a smooth but
possibly stiff transition. Neither is "the differentiable one" any more; pick the
one that integrates better for your model.

### The catch (it's a modeling choice, not a free lunch)

The sigmoid has a **sharpness** `s_sharp`. Turn it up and you approach a true
step — but the ODE stiffens and the sensitivities near `tau` steepen. Turn it down
and it integrates easily but looks less like a step. And because data rarely
constrains how sharp the switch is, `s_sharp` is usually **fixed** (a modeling
assumption) while the physical parameters — and `tau` — are fitted. (Other smooth
steps work too: `tanh`, `arctan`, a Gaussian CDF/`erf`, a Hermite smoothstep.)

## Takeaway

Don't reach for a metaheuristic just because a model has a switch in it. A
discontinuity in **time** — a dose, a stimulus, a treatment window — is
gradient-fittable, switch time included. What actually rules the gradient path out
is a prediction with no derivative at all: a **stochastic** simulation (`ssa`,
NFsim). For those, use `de`/`pso`/`ss`/…

> **This lesson used to say the opposite.** Through bngsim 0.12.1 the hard `if()`
> model was refused by `trf`/`lbfgs` — the forward-sensitivity engine had no
> support for the construct — and the lesson was titled "when a gradient fit is
> refused, and how to fix it", with smoothing as the fix. bngsim 0.12.2
> differentiates it, `tau` and all. The advice that survived is narrower and more
> durable, which is why the refusal example moved to `ssa` rather than being
> deleted.
