# Lesson 47 — One model, many conditions

**Feature:** fit a wildtype *and* a perturbed (knockout) condition of the same system with **one** model file, via `condition:` perturbations · **Difficulty:** ★★ · **Tier:** recovery

You measured the same reaction in two biological conditions — wildtype cells and a
knockout that ablates one reaction — and you want to fit both at once. The old way
was to copy the model file once per genotype (`reversible.bngl`, `knockout.bngl`, …)
and fit them as separate models. The PEtab-aligned way — edition-2 **Mechanism A** —
keeps **one** model and names each condition as a `condition:` perturbation, then
binds each experiment to its condition.

## The model

A [reversible conversion](reversible_conversion.bngl) `A ⇌ B` (forward rate `kf`,
reverse `kr`, both **fitted**). It relaxes to an equilibrium set by the *ratio* of
the rates:

```
dB/dt = kf·A − kr·B          B_eq = kf/(kf+kr) · A0        relaxation rate = kf + kr
```

- **Wildtype** — both rates active. `Obs_B` relaxes to `B_eq`, so a single time
  course sees *both* `kf + kr` (the rate) and `kf/(kf+kr)` (the equilibrium ratio).
- **Knockout** — the reverse reaction is ablated (`kr` set to 0). `Obs_B` now runs
  irreversibly to the full total `A0` at rate `kf` — a clean second look at `kf`.

## One model + a condition, in one conf

```
condition: ko, perturbations: kr = 0        # the reverse-reaction knockout
experiment: wildtype,               data: wildtype.exp   # no condition → the reaction as written
experiment: knockout, condition: ko, data: knockout.exp  # kr forced to 0
uniform_var = kf  0.05  3.0
uniform_var = kr  0.02  2.0
```

A `condition:` is a **named set of perturbations** on the one model — here a single
absolute override `kr = 0`. Because it is *absolute*, it does not touch the fitted
`kr` that still drives the wildtype simulation; it only pins `kr` to 0 in the
knockout's own simulation. Each `experiment:` then binds to its condition (or to
none, for the wildtype). Both experiments share the fitted `(kf, kr)`, and running
the conf recovers both rates on the nose — which is what
[`tests/test_tutorial_examples.py`](../../../tests/test_tutorial_examples.py) checks.

## Only the diagonal is simulated

With **N** experiments and **M** conditions on one model there are `N × (M+1)`
possible `(experiment, condition)` combinations, but only the **scored diagonal** —
each experiment under *its own* condition — is ever used. PyBNF simulates only that
diagonal:

| | *(no condition)* | `ko` |
|---|---|---|
| **`wildtype`** | ✅ scored (`wildtype`) | ✗ never run |
| **`knockout`** | ✗ never run | ✅ scored (`knockoutko`) |

The off-diagonal cells (`wildtype`-under-`ko`, `knockout`-under-nothing) are never
scored, so they are never computed. For this 2×2 that is 2 simulations per objective
evaluation instead of 4; on a real genotype panel (say 5 experiments × 4 conditions)
it is the difference between 5 and 25 (lanl/PyBNF [#484](https://github.com/lanl/PyBNF/issues/484)).

## Run it

```bash
pybnf -c condition_perturbations.conf
```

## Where this sits

- [Lesson 16](../16_joint_fit) — a joint fit across multiple *model files*; this is a
  joint fit across conditions of **one** model (the PEtab-aligned alternative).
- [Lesson 30](../30_data_fusion) — several *experiments* on one model, but no
  `condition:` perturbations (one shared condition).
- [Lesson 9](../09_experiment_design) / [Lesson 29](../29_petab_protocols) —
  conditions applied *inline* as pre-equilibration (a two-phase protocol), rather
  than as the separate measured simulations here.

## Regenerating the data

```bash
python ../regenerate_data.py 47_condition_perturbations
```

`wildtype.exp` is the model's `Obs_B` at the true rates; `knockout.exp` is `Obs_B`
generated **under the `ko` condition** (`kr = 0`), so it is the knockout curve
(rising to `A0`), not the wildtype. Truth lives test-side in
[`_manifest.py`](../_manifest.py).
