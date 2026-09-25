# Each declared experiment starts from the model as written, so the action list restores the parameters and the seed species before every experiment, and their order cannot change a fit

## Status

Accepted and implemented (2026-09-25). Fixes #830, #831, #869 and #875. It changes the action
list PyBNF writes for a BNGL model's experiments (`pybnf/pset.py`), the bngsim `.net` bridge
(`pybnf/bngsim_model/net_model.py`), the bngsim network-free bridge (`nf_model.py`) and the
bridge classifier (`classification.py`). The decision is the maintainer's: "Order shouldn't
matter."

## The problem

PyBNF writes every experiment of a BNGL model into one BNGL action list, and a BNGL action
carries its effect forward to the actions after it. Between experiments PyBNF wrote only
`resetConcentrations()`, and not even that on the network-free path. So an experiment could
start from state that an earlier experiment left behind:

- **Parameters.** A pre-equilibration's inline condition (`setParameter("Stimulus_isOn",0)`)
  stayed in force for every experiment written after it. BNG2.pl's `parameter_scan` never
  restores the scanned parameter, so everything after a dose scan ran at its last dose; bngsim
  restores it, so the two backends disagreed (#831).
- **Species.** A pre-equilibrated scan saved its post-intervention state with an unlabelled
  `saveConcentrations()`, which redefines what every later `resetConcentrations()` restores.
  The next experiment started from that snapshot, not from the seed species (#830).
- **Condition runs on bngsim.** A `condition:` experiment runs as a mutant, and its engine
  was cloned from the base run's engine after the base run's actions had changed it (#869).
- **Network-free.** BioNetGen's `simulate_nf` reads its final state back into the model, and
  the bngsim network-free bridge kept one live session for the whole action list, so a second
  `method: nf` experiment continued from the molecules the first left behind (#875).

In tutorial lesson 9's three experiments (a washout, a dose response and a growth curve),
only 2 of the 12 combinations of declaration order and backend scored the closed form.

## The decision

1. **Every synthesized experiment starts with the experiment start**
   (`BNGLModel._append_experiment_start`): `saveParameters("pybnf_experiment_start")` before
   the first experiment, `resetParameters("pybnf_experiment_start")` before every later one,
   and then `resetConcentrations()`, on the network-free path too. The saved parameters are the
   model's as written, with the trial point's free parameters (and, in a condition run, its
   mutation). The save comes after any hand-written action, so an edition-1 `begin actions`
   block that sets a parameter before a legacy `time_course` still sets it for every
   synthesized experiment. A label keeps these lines apart from any `saveParameters()` the
   model's own actions use.
2. **A pre-equilibrated scan records its state under its own label**,
   `saveConcentrations("<name>_scan_start")`, so it cannot become what a later
   `resetConcentrations()` restores. The scan itself restarts each dose from the state at its
   invocation on both backends (BNG2.pl saves it under its own `SCAN` label; bngsim's native
   scan captures its live state), so the save is a record, not an input. It also keeps a
   network-free pre-equilibrated scan off the network-free bridge, which has no species
   snapshot store and would start every dose from the seed.
3. **The bngsim bridges keep BioNetGen's snapshots.** Both used to read every save/reset line
   as the default slot. Now `saveConcentrations("x")` / `resetConcentrations("x")` use
   bngsim's labelled snapshots and leave the default one alone, `saveParameters("x")` /
   `resetParameters("x")` keep a snapshot per label, and a labelled reset with no save is
   refused by name, as BioNetGen stops on it. An argument that is not one quoted label is
   refused rather than read as the default. `resetParameters` restores a derived parameter
   as BioNetGen does, as its expression: primaries are written first, then each derived
   parameter that tracked its expression is re-attached (a `setParameter` on it had overridden
   the expression, bngsim #188), and a parameter that cannot be restored raises.
4. **The network-free bridge runs `resetConcentrations()`** by ending its live session; the
   next action starts a fresh one from the seed species under the parameters then in force.
   Nothing can have redefined its default snapshot, because `saveConcentrations` stays
   classified network-only. A labelled `resetConcentrations("x")` is refused there. The
   classifier now routes `resetConcentrations()`, `saveParameters` and `resetParameters` to
   both bridges.
5. **A condition run on bngsim starts from the engine as `execute` received it** (#869): a
   clone taken before the base run's actions, from which each condition's model is cloned.
   A relative perturbation of a non-free target reads its base value from that clone too. This
   also covers an edition-1 job whose hand-written block sets a parameter before a `mutant =`
   line, which no reset in the synthesized list could reach.

Only the synthesized list changes. A hand-written actions block keeps BNG2.pl's semantics:
each action starts from the state the previous one left, and a network-free block with no
reset still continues its state. Within one pre-equilibration experiment nothing changes: its
equilibrated state and its conditions carry into its measured phase.

## Checked, path by path

- **BNG2.pl** (`BNGLModel`, `NetModel`): runs the new lines natively. `saveParameters`
  keeps `Constant` and `ConstantExpression` parameters (`ParamList::copyConstant`), so a
  derived parameter reads its primaries again after the reset.
- **bngsim `.net`**: decisions 3 and 5. The gradient path runs the same list: after the resets
  a fresh experiment's sensitivities are seeded from the seed and a washout's from its
  equilibration, and each experiment's dA/dk matches its closed form in every order.
- **bngsim network-free**: decision 4.
- **SBML and Antimony** (`bngsim_sbml_model.py`) build every experiment's engine afresh from
  the template, and RoadRunner resets before every action, undoes each mutation and restores a
  scanned parameter, so neither carries state between experiments. Unchanged.
- **Multiple shooting** (`job_type = ms`) builds each lane from the declared parameters and
  seed species and replays none of the list's lines, so it already treated each experiment
  as independent (its separate problem, replaying none of an experiment's own protocol, is
  #910). Its `.net` lane now passes its never-run engine to `_get_mutant_model_bngsim`.
- **Emit-set pruning** (ADR-0069) assumed each synthesized experiment was reset-independent;
  now it is.

## Consequences

- **Stochastic runs draw different streams.** The seed policy derives each stochastic
  action's seed from its position in the list, and the new lines shift every position, so an
  SSA or network-free edition-2 job gives different (equally distributed) trajectories than
  before for the same `random_seed`. On `examples/real-world/Kozer-2013/egfr_nf`, with the
  seed made independent of the position, the objective is identical before and after.
- **The edition-2 IGF1R example changes.** In `examples/real-world/Erickson-2019/igf1r`,
  `F5D_60min`'s pre-incubation started from `F5D_20min`'s post-wash snapshot, off by up to 321
  of 335 counts; it now equals a job that declares `F5D_60min` alone, and the objective at the
  default point falls from 9.97e8 to 3.89e8. Its `VALIDATION.md` fit used the edition-1
  `igf1r_legacy.conf`, which is unaffected. The other deterministic edition-2 examples with
  several experiments on one model (tutorials 30 and 47, `Kozer-2013/egfr_ode`,
  `Salazar-Cavazos-2019/egfr_simpull`) give the objective they gave before.
- **Load-time refusals that exist only because of #830** can go. The `perturbations: none`
  refusal proposed for #906 (ADR-0150) and its strict expected failure are among them.

## Alternatives rejected

- **One BNG2.pl run per experiment.** Correct by construction, but it multiplies network
  loading and process start-up by the number of experiments on every evaluation.
- **A labelled seed snapshot in place of `resetConcentrations()`.** The default snapshot is
  already the seed as long as nothing synthesized redefines it, which decision 2 ensures. A
  labelled seed would add a line, and a second meaning of "the seed" (a snapshot taken at one
  moment rather than the seed species at the current parameters) that the two backends would
  then have to agree on.
- **Dropping the scan's save altogether.** The scan does not read it, but without a
  network-only line a network-free pre-equilibrated scan would route to the network-free
  bridge and silently start every dose from the seed.
- **A trailing `resetParameters` after the last experiment, to fix #869.** It would not
  reach an edition-1 hand-written `setParameter` before a `mutant =` line.
