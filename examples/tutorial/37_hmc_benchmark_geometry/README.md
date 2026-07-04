# Lesson 37 — HMC / NUTS on benchmark geometries

**Feature:** `job_type = hmc` (blackjax NUTS) on the built-in analytical target menu · **Difficulty:** ★★☆

PyBNF's Hamiltonian Monte Carlo sampler (`job_type = hmc`, blackjax NUTS) samples
a posterior using its **gradient**. It runs only on a **JAX-traceable analytical
target**: either a built-in benchmark distribution (this lesson) or your own
`objective = expression` (lesson 38, where the target is a closed-form ODE
likelihood). It cannot sample a simulated BNGL/SBML model — the solver is not
differentiable in-graph.

This lesson introduces the sampler on targets whose shape you already know, so
you can see what "hard" and "easy" look like before putting HMC on a real model.

## The built-in menu

You name a target directly on the `objective` line — no model, no data:

```
gaussian, rotated_gaussian, banana, multimodal, rotated_quartic
```

The target's coordinates are declared as free parameters (with wide flat bounds),
and HMC samples them.

## The two examples

| Conf | Target | Geometry | Divergences? |
| --- | --- | --- | --- |
| [`gaussian.conf`](gaussian.conf) | `gaussian, mean = 0 0, variance = 1 1` | round, uncorrelated — trivial | none |
| [`banana.conf`](banana.conf) | `banana, a = 1, b = 100` | Rosenbrock crescent — strong curvature | only if under-tuned |

The **banana** (Rosenbrock) has log-density `−[(a−x1)² + b·(x2−x1²)²]`: a long,
curved valley along `x2 = x1²`. It is the classic sampler stress test — a random
walk crawls along the valley, while HMC uses the gradient to follow it. The
density concentrates near `(x1, x2) ≈ (1, 1.5)`.

```
pybnf -c gaussian.conf
pybnf -c banana.conf
```

## The two knobs on a curved target

On the banana, the two settings that matter are:

- **`num_warmup`** — adapts the step size and mass matrix before sampling.
- **`target_accept`** — how cautious the leapfrog integrator is (higher ⇒ smaller
  steps, safer on curvature).

Too small a warmup or too low a `target_accept`, and NUTS produces **divergent
transitions** — PyBNF prints a warning that the draws are not a trustworthy
reference. `banana.conf` uses a long warmup and `target_accept = 0.95` so the
sampler negotiates the bend cleanly; lower them and watch the warning appear.
`gaussian.conf` needs neither — its geometry is trivial.

Run the two back to back: same sampler, same budget, wildly different geometries.
That gap is why lesson 38 reaches for gradient-based sampling on real (correlated)
model posteriors.

## Notes

- HMC needs the optional extra: `pip install pybnf[jax]` (jax + blackjax).
- Posterior draws are written to `output/.../Results/samples.txt`; set
  `output_inference_data = 1` (with `pip install pybnf[arviz]`) for an ArviZ
  `InferenceData` with trace/pair plots and R-hat / ESS diagnostics.
- These targets are pure benchmark distributions. To put HMC on a real model
  whose likelihood is a closed-form ODE solution, see **lesson 38**.
