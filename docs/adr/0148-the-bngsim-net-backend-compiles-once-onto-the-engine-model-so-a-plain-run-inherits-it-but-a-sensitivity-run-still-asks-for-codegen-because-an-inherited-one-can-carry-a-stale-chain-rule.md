# The bngsim `.net` backend builds its compiled right-hand side once onto the engine model, so a plain run inherits it, while a sensitivity run still asks for codegen, because an inherited sensitivity artifact can carry a stale chain rule

## Status

Accepted and implemented (2026-09-25). It changes how often, and through which bngsim calls,
the bngsim `.net` backend (`pybnf/bngsim_model/net_model.py`) compiles an ODE right-hand side.
It does not change a simulated number: the compiled right-hand side it uses is the one it
used before, and every oracle below is a closed form.

## The problem

`BngsimModel` asked bngsim for a compiled right-hand side in three places, each costlier than
it needed to be.

- **Every per-evaluation copy re-loaded the model.** `copy_with_param_set` and
  `_get_mutant_model_bngsim` make a shallow copy with `copy.copy`. `BngsimModel` defines
  `__getstate__`/`__setstate__` for Dask, and `copy.copy` goes through them, so every copy
  re-loaded the `.net` file with `Model.from_net` and re-ran codegen. The caller then replaced
  that engine model with a clone of the original. The load and the codegen call were thrown
  away, once per evaluation.
- **Every `Simulator` asked for codegen.** `_codegen_kwargs` returned
  `{'codegen': True, 'net_path': ...}` for every ODE construction. On bngsim 0.15.0 that went
  through bngsim's `.net` codegen path, whose in-process memo made it nearly free. bngsim
  lanl/bngsim#803 routes `.net` models through the path SBML models take, whose cache key is
  computed from the model's structure on every `codegen=True` request, and that costs time
  proportional to the model (lanl/bngsim#820).
- **The flag came from a deprecated call.** Construction and unpickling called
  `bngsim.prepare_codegen(net_path)`, used only as a "codegen is available" flag. Since
  lanl/bngsim#803 it loads the file again, derives a Jacobian, computes the key, and warns
  `DeprecationWarning`, as does `Simulator(net_path=...)`.

Measured per evaluation (one `copy_with_param_set` plus one action's `Simulator`), with a
warm codegen cache:

| model | reactions | bngsim 0.15.0, before | after | bngsim with #803, before | after |
|---|---|---|---|---|---|
| `examples/egfr_benchmark/egfr_ground.net` | 3,749 | 9.5 ms | 1.4 ms | 195 ms | 1.4 ms |
| `examples/fceri_gamma/fceri_gamma2.net` | 58,276 | 121 ms | 14 ms | 3.0 s | 14 ms |

What remains after the change is `Model.clone()` itself. On a function-heavy model (400
Functional rate laws), an action's construction also fell from 13.5 ms to 2.5 ms on 0.15.0,
because clones now inherit the engine model's derived analytical Jacobian instead of each
deriving its own.

## The decision

1. **Build once, onto the engine model.** `_attach_codegen` constructs one
   `Simulator(engine_model, method='ode', codegen=True)` at construction and on unpickle.
   bngsim records the artifact on the model a `Simulator` was built for, and `Model.clone()`
   carries it. This uses only public behaviour, present in 0.15.0 and in #803.
2. **A copy clones the engine model.** `BngsimModel.__copy__` is a shallow copy that takes
   its own `Model.clone()` of the engine model, and the clone carries the artifact and the
   derived Jacobian. Both callers used to clone right after copying. The clone now happens
   in one place, so no copy can share its parent's engine model, and a `set_param` on it.
   Pickling still goes through `__setstate__`, which re-loads and rebuilds, once per Dask
   worker, since models are scattered with `broadcast=True`.
3. **A plain run inherits.** `_codegen_kwargs` returns `{}`, and the `Simulator` takes the
   artifact the clone carries, without recomputing the key. A plain right-hand side reads
   parameter values at run time, so one artifact serves every clone, whatever a mutant
   overrides.
4. **A sensitivity run asks for codegen.** `_codegen_kwargs` returns `{'codegen': True}` for
   every construction that carries sensitivities, so bngsim rebuilds against the model as
   it stands. That holds whether or not the plain build succeeded, and under
   `PYBNF_NO_CODEGEN`: a sensitivity right-hand side is compiled or there is none. The one
   exception is `BNGSIM_NO_CODEGEN`, where it passes nothing and bngsim refuses the run with
   its own explanation; `codegen=True` would have it build anyway. Whether a construction
   carries sensitivities is the construction's, not the model's: an unscored carried-state
   scan drops the model's request, and gets a plain run's answer. See below for why a
   sensitivity run must not inherit.
5. **No artifact, no retry.** When codegen is disabled (`PYBNF_NO_CODEGEN`,
   `BNGSIM_NO_CODEGEN`) or its one build failed, a plain run passes `codegen=False`. bngsim then
   neither retries a failed build on every construction (lanl/bngsim#826) nor, since #803,
   compiles a model of 256 or more species on its own.

## Why a sensitivity run does not inherit

An inherited *sensitivity* artifact can describe a different model from the one it runs.
In every evaluation with conditions, the base condition's sensitivity `Simulator` is built on
the evaluation's engine model and attaches its artifact there, and each condition's model is
cloned from that engine model afterwards. `analytic_sens_rhs_status` does the same to the
base model, so every later copy carries it. A condition that overrides a derived parameter
changes which primaries reach which rate laws. The inherited artifact keeps the base
condition's chain rule, and bngsim does not check (lanl/bngsim#708). (A multiple-shooting lane
builds on a clone of its own, from `_get_mutant_model_bngsim`, and attaches nothing to the
base model.)

This does not depend on the plain build. Under `PYBNF_NO_CODEGEN`, or after a failed build,
the base condition's sensitivity run still compiles and attaches an artifact. The first
version of this change passed nothing to a sensitivity run with no plain artifact, and so let
each condition inherit the base condition's; the independent review caught it
(`test_gradient_under_pybnf_no_codegen_is_not_served_another_conditions_chain_rule`).

Measured on `k2 = 2*k1` driving `A -> B`, with `k2` pinned to 5 on the clone, so the closed
form is `dA/dk1 = 0`:

| sensitivity run's codegen request | max \|dA/dk1\| |
|---|---|
| `codegen=True` (this decision) | 0.0000 |
| none, inheriting the probe's artifact | 1.4325 |

`test_a_sensitivity_run_is_not_served_a_stale_chain_rule` pins this. On bngsim 0.15.0 it is a
strict expected failure: there even a fresh build keeps the stale chain rule, through bngsim's
`.net` codegen path (lanl/bngsim#694), which #803 removes.

## Alternatives rejected

- **A memo in bngsim's `prepare_codegen`, keyed on the file's contents.** It would make the
  flag call cheap again. But it brings back the file-keyed cache #803 removes, and it leaves
  both the per-copy re-load and the per-`Simulator` key in place.
- **A sensitivity-shaped warm-up, as the SBML backend does (#543).** It would spare a gradient
  fit the key on each sensitivity construction, and it inherits a sensitivity artifact, which
  is the hazard measured above.
- **Keep `codegen=True` on every construction, and fix only the copy.** Correct, but on bngsim
  with #803 each plain run then pays the key: 89 ms at 3,749 reactions and 1.4 s at 58,276,
  per action per evaluation.

## Consequences

- A gradient fit on bngsim with #803 still pays the key on each sensitivity construction.
  That is the price of item 4, and it falls with lanl/bngsim#820 (the key computed in C++) or
  lanl/bngsim#708 (a reused artifact checked against the model).
- This change should reach users before, or with, the bngsim release that carries #803.
  Without it, a fit on a large `.net` model would slow from 121 ms to 3.0 s per evaluation at
  58,276 reactions.
- The SBML backend's sensitivity-shaped warm-up (#543) inherits a sensitivity artifact too.
  Whether a condition or mutant there can override a derived quantity, and so meet
  lanl/bngsim#708, was not examined here.
