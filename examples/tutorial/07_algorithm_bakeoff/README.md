# Lesson 7 — An algorithm bake-off (six optimizers, one hard landscape)

**Feature:** the metaheuristic optimizer family (`de`/`ade`/`pso`/`cmaes`/`sa`/`ss`) · **Difficulty:** ★★☆

Every earlier lesson picked one optimizer and moved on. This one fixes the
*problem* and varies the *optimizer* — six of them — so you can see that PyBNF's
whole gradient-free family solves the same fit, and get a feel for how each one
searches.

## The landscape

The model ([`oscillator.bngl`](oscillator.bngl)) is a linearized Lotka-Volterra
predator-prey oscillator:

```
dP/dt = -alpha * Q
dQ/dt =  gamma * P        →   undamped oscillation at frequency w = sqrt(alpha*gamma)
```

We fit `alpha` and `gamma` from the two time courses. This is a *deliberately
awkward* target: the error as a function of `(alpha, gamma)` is **oscillatory**,
because a trial frequency that is slightly wrong drifts in and out of phase with
the data over the time window, carving the surface into ridges and local minima at
nearby and aliased frequencies. A naive local optimizer would fall into the first
dip it found. A global search has to climb back out — which is exactly what these
six algorithms are built to do.

## The entrants

Each `.conf` is identical except for its `job_type` and search budget:

| Conf | `job_type` | Strategy in one line |
| --- | --- | --- |
| [`oscillator_de.conf`](oscillator_de.conf) | `de` | population + difference-vector mutation (the robust default) |
| [`oscillator_ade.conf`](oscillator_ade.conf) | `ade` | the same, but asynchronous — updates as each sim finishes |
| [`oscillator_pso.conf`](oscillator_pso.conf) | `pso` | a swarm with momentum, pulled toward personal + global bests |
| [`oscillator_cmaes.conf`](oscillator_cmaes.conf) | `cmaes` | a Gaussian that adapts its full covariance to the valley's shape |
| [`oscillator_sa.conf`](oscillator_sa.conf) | `sa` | a single walker that accepts uphill moves while "hot", then cools |
| [`oscillator_ss.conf`](oscillator_ss.conf) | `ss` | a diverse reference set, combined + locally polished each iteration |

Run the whole bake-off:

```bash
for m in de ade pso cmaes sa ss; do pybnf -c oscillator_$m.conf; done
```

All six recover `alpha ≈ 1.2` and `gamma ≈ 0.8` (to within 3%, asserted in
[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py)) —
the point of the lesson: the whole metaheuristic family is interchangeable on the
*outcome*, and you choose among them on *character*.

## What differs in practice

- **`de` / `ade` / `pso`** are the general-purpose workhorses — few knobs, hard to
  go wrong. `ade` shines on a cluster (no idle workers waiting on a slow
  generation).
- **`cmaes`** is usually the most *sample-efficient* here, because it learns that
  `alpha` and `gamma` trade off along the frequency `sqrt(alpha*gamma)` and moves
  along that diagonal instead of across it.
- **`sa`** is the only single-walker method; its willingness to step *uphill* while
  hot is what frees it from a wrong-frequency minimum.
- **`ss`** does the most work per iteration (it polishes every combination with a
  built-in local search), so its budget is set much smaller than the others' — a
  reminder that "population_size × max_iterations" is not comparable across
  algorithm families.

## The finishing polish

Every conf ends with `refine = 1`: after the global search narrows down the basin,
a local Simplex polish drops the objective the last few orders of magnitude onto
the true optimum. Global search to *find* the basin, local search to *nail* it — a
pattern worth reusing (see Lesson 3).

## Addendum: Powell, the local contrast

The six above are all *global* metaheuristics. It's worth seeing the other kind on
the same landscape — a **local** optimizer, **Powell** (`job_type = powell`), a
derivative-free conjugate-direction method (a cousin of Simplex). A local
optimizer is exactly the "naive" method the landscape section warned about: it
starts from a **single point** and descends into the nearest dip.

So a start-point optimizer takes an initial **guess** per parameter with `var`
(one value), not a search range with `uniform_var`:

| Conf | start | outcome |
| --- | --- | --- |
| [`oscillator_powell.conf`](oscillator_powell.conf) | `var = alpha 1.3, gamma 0.9` (right basin) | walks straight to `(1.2, 0.8)`, cheaply |
| [`oscillator_powell_trapped.conf`](oscillator_powell_trapped.conf) | `var = alpha 3.0, gamma 3.0` (wrong basin) | **trapped** at an aliased frequency |

From a guess already in the right basin, Powell nails the truth in far fewer
evaluations than any global search. From a guess in the wrong basin it gets stuck —
it has no way to climb back out. That is the whole reason the bake-off uses global
methods: they don't need a good starting point. Powell (like Simplex) earns its
keep as a fast **refiner** *after* a global search has found the basin (it's a
valid `refine_method`), or when you already have a solid initial estimate.

```bash
pybnf -c oscillator_powell.conf           # good start -> recovers
pybnf -c oscillator_powell_trapped.conf   # bad start  -> traps
```
