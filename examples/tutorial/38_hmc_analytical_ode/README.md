# Lesson 38 — An analytical ODE solution as an HMC likelihood

**Feature:** `objective = expression` + `job_type = hmc` (blackjax NUTS) on a closed-form ODE · **Difficulty:** ★★★

Some ODE models have an exact **closed-form solution**. When they do, parameter
estimation needs no simulator at all: the solution `y(t; θ)` is written directly
as the model prediction, the Gaussian log-likelihood becomes a plain algebraic
expression, and PyBNF's gradient-based sampler (`job_type = hmc`) differentiates
it and samples the posterior with Hamiltonian Monte Carlo / NUTS.

This lesson does exactly that on two catalog models with known analytical
solutions.

## The idea in one line

An ODE with a closed form:

```
dV/dt = -c·V        ⇒        V(t) = V0·exp(-c·t)
```

becomes the whole "model." There is no `.bngl`, no network generation, no ODE
solve — just the differentiable expression `V0*exp(-c*time)`.

## How the likelihood is assembled

`objective = expression` turns a math expression into a **per-observation
negative log-likelihood**. `data = viral_decay.exp` binds the data file; its
column headers (`time`, `Vobs`) join the free parameters (`V0`, `c`) as symbols.
PyBNF evaluates the expression once per data row and **sums** it:

```
expression = 0.5 * ((Vobs - V0*exp(-c*time)) / sigma)^2
```

That is the Gaussian sum-of-squared-residuals NLL (with `sigma` fixed at the
known noise level, so its constant term drops out). Because every symbol is
differentiable, HMC obtains an exact gradient of the log-density.

The sampler assembles `log p(θ | data) = log p(θ) − objective`, where `log p(θ)`
comes from the parameter priors: a flat `uniform_var` box gives the
maximum-likelihood / least-squares estimate, while `normal_var` / `lognormal_var`
gives a full Bayesian posterior.

## Why HMC (and what it refuses)

`job_type = hmc` needs a **JAX-traceable analytical target** — an `expression`
here, or a built-in menu target (lesson 37). It **refuses** a simulated
BNGL/SBML model, whose solver is not differentiable in-graph, with a pointed
error. The payoff is a No-U-Turn sampler that follows the posterior's gradient,
mixing far more efficiently than a random walk and returning full posterior
draws rather than a single point estimate.

## The two examples

| Conf | Closed form | Fits | Posterior |
| --- | --- | --- | --- |
| [`viral_decay_hmc.conf`](viral_decay_hmc.conf) | HIV/ART decay `V0·e^{−c·t}` | `V0`, `c` | well identified; mild `V0`–`c` correlation |
| [`damped_oscillator_hmc.conf`](damped_oscillator_hmc.conf) | damped oscillator `C·e^{−a·t}·cos(ω·t)` | `C`, `a`, `ω` | correlated (`C`–`a` ridge) — where NUTS earns its keep |

```
pybnf -c viral_decay_hmc.conf
pybnf -c damped_oscillator_hmc.conf
```

Both recover the truth the data was generated from (V0≈100, c≈0.6; C≈5, a≈0.35,
ω≈3). Posterior draws land in `output/.../Results/samples.txt`.

## Reading the posterior in ArviZ

Set `output_inference_data = 1` (needs `pip install pybnf[arviz]`) to also emit
an ArviZ `InferenceData`, which loads straight into ArviZ for trace / pair /
posterior plots and R-hat / effective-sample-size diagnostics. The
`damped_oscillator` pair plot shows the tilted `C`–`a` ridge that motivates
gradient-based sampling.

## Regenerating the data

The `.exp` files are synthesized from the closed forms (not a simulator) by
[`regenerate_fixtures.py`](regenerate_fixtures.py) — the same solution that
serves as the likelihood also generates the data, which is what makes these
clean recovery demonstrations.

## Notes

- The models come from the analytical-ODE catalog (`hiv_art_viral_decay`,
  `damped_harmonic_oscillator`). Only their closed forms are used here.
- The PEtab-math expression grammar supports `exp`, `cos`, `sin`, `sqrt`,
  powers (`^`), etc. — all lowered to differentiable JAX for the HMC gradient.
- See lesson 37 for HMC on the built-in benchmark geometries (the banana), and
  the note there on `num_warmup` / `target_accept`.
