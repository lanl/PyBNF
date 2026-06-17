# The algorithm layer asks the parameter for the θ↔u transform; the inverse PSet bridge joins `_param_vec` (issue #412)

ADR-0010 settled that `FreeParameter` owns a first-class `Scale` so the
`log10`/`10**` boundary "lives in exactly **one** place." That held *inside*
`FreeParameter` (`add` / `set_value` / `_reflect` / `prior_logpdf` /
`sample_value` / `value_from_quantile`), but the **algorithm layer bypassed the
`Scale` object** and inlined the transform directly — `np.log10(x) if v.log_space
else x`, `10 ** u if v.log_space else u`, `exp10(...)` — at ~12 sites across
`base._param_vec`, the start-point optimizers (`local_base`, `powell`), `simplex`
(the densest cluster: centroid / reflection / contraction / shrink / degeneracy /
`get_sums`), `particle_swarm`, the samplers' histogram path, `adaptive_mcmc`'s
chain state, and `dream`'s `u→θ`. So the θ↔u boundary was **replicated, not
centralized** — the drift/bug surface ADR-0003 warns against, and the concrete
blocker that made the `LogBase(n)` (second log base) question untenable. This is
a **behavior-preserving** cleanup (the contract is the existing green optimizer +
sampler tests, kept byte-green — mirroring the M2.3 `Prior` and M2.4 `NoiseModel`
extractions). We settled this shape:

- **`FreeParameter` exposes a public θ↔u pair** — `to_sampling_space(θ)` (=
  `_scale.forward`) and `from_sampling_space(u)` (= `_scale.inverse`), the public
  peers of the private scale methods. The algorithm layer **asks the parameter**
  for the transform instead of inlining `log10`/`10**`. `to_sampling_space`
  accepts a scalar or a numpy array (the histogram path passes a whole column).
  `from_sampling_space` is the **unguarded**, bit-for-bit `10.0 ** u` the proposal
  arithmetic already relied on (pinned by `test_priors`).

- **The inverse PSet bridge `_pset_from_u` is hoisted onto `Algorithm`**, next to
  the existing forward bridge `_param_vec`, so the u-vector↔PSet conversion lives
  in one place too. It takes a `reflect` flag: `reflect=True` folds an out-of-box
  coordinate into the box (the start-point optimizers), `reflect=False` lets the
  offending `set_value` raise `OutOfBoundsException` so the caller can **reject**
  the proposal (DREAM). This removes the copy in `local_base`; `dream`'s
  `_proposal_pset` and `local_base`'s `_u_from_pset`/`_pset_from_u` alias pair now
  route through it.

- **Simplex's internal arithmetic builds u-vectors via the transform pair and
  maps back**, collapsing each per-operation `if v.log_space` branch into one
  scale-agnostic expression (`from_sampling_space(... to_sampling_space(a) ...)`).
  Linear is the identity, so the two former branches fold into one. The centroid —
  formerly the *guarded* `exp10` — now uses the unguarded `from_sampling_space`,
  making it consistent with its sibling reflection/expansion/contraction ops,
  which already used a bare `10**` immediately clamped to the box.

- **`exp10` stays as the guarded inverse for user-supplied start values.** The two
  start-point parse sites (`simplex._parse_start_point`,
  `local_base._resolve_start_pset`) keep `exp10(v.p1) if v.log_space else v.p1`,
  because `exp10` raises a **configuration hint** on overflow ("you declared a
  `lognormal_var`/`logvar` and specified the arguments in regular space instead of
  log10") — a user-facing guard, tested via `algorithms.exp10`. The split is
  principled: a **user-supplied** config value (a `logvar` start `p1`) is guarded;
  an **algorithm-generated** u-vector is not (an out-of-range coordinate is clamped
  or reflected at the call site, never an error). `from_sampling_space` is the
  unguarded transform precisely so a mid-fit proposal cannot crash the run.

## Considered Options

- **Route the start-point parse through `from_sampling_space` too (delete the
  `exp10` branch).** Rejected: it silently turns a misconfigured `logvar` `p1`
  into `inf` (an unbounded `var`/`logvar` accepts it) instead of the helpful
  overflow error — the guard exists for exactly that common mistake and is tested.
- **Add the overflow guard to `from_sampling_space` / `Scale.inverse`.** Rejected:
  the proposal arithmetic relies on the unguarded `10**u` followed by a box clamp;
  raising mid-fit would crash a run, and `test_priors` pins `Scale.inverse ==
  10.0 ** u` bit-for-bit (the single source of truth the public method must equal).
- **Leave `_pset_from_u` in `local_base`.** Rejected: DREAM and Simplex need the
  same conversion; the inverse bridge is the ≥2-user event that earns the hoist
  next to `_param_vec` (ADR-0009), and one place is the whole point of #412.

Relevant ADRs: **0003** (prior and proposal share one scale), **0010** (scale
lives in one place — the intent this restores at the algorithm layer), **0009**
(the ≥2-user hoist that already put `_param_vec` on `Algorithm`).
