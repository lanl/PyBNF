# An edition-2 model's actions may be only its network definition, because the model file defines the model and the conf defines the protocol

## Status

Accepted and implemented (2026-09-25). Implements #969 and fixes #963 item 3. The decision is
the maintainer's (#969): in edition 2, the model file defines the model, and the conf (or the
PEtab tables) defines the protocol. It changes the BNGL scanner (`BNGLModel` in
`pybnf/pset.py`), config load (`config._load_experiments`), the PEtab exporter
(`clean_model_for_petab` in `pybnf/petab/export.py`, whose #900 whitelist it replaces) and the
PEtab importer (`import_job` in `pybnf/petab/import_.py`).

## The problem

In an edition-2 job a BNGL model's own actions, in a `begin actions` block or loose after
`end model`, and the conf's `experiment:` lines were two sources of protocol. PyBNF ran both:
first every hand-written action, then the simulations it builds from the `experiment:` lines,
spliced onto the end of the same BNGL action list. Nothing defined what that splice meant, so
whether a hand-written action changed a fit depended on what came after it and on what kind of
state it changed:

- a leftover `simulate` was undone by the `resetConcentrations()` before a network-based
  experiment, but before ADR-0151 it carried into a network-free experiment, which got no
  reset (#875);
- a `setParameter` persists into every experiment: ADR-0151 saves the experiment-start
  parameters after the hand-written actions, so it restores the value the actions set (#830);
- a hand-written `parameter_scan` leaves its parameter at the last scanned value on BNG2.pl
  and puts it back on bngsim, so the same job fits differently on the two backends (#941);
- a `saveConcentrations()` redefines the snapshot every later `resetConcentrations()`
  restores.

Each of these was patched where it surfaced, and the PEtab exporter carried a list of the
actions it could drop without changing the fit (#900, PR #944), refusing the rest. The list
had to reason about every action, every experiment method and both backends, and the fitter
itself still ran whatever the model held. The importer copies a third-party BNGL model byte
for byte, so it could bring such actions into an imported job too.

## The decision

1. **An edition-2 model's actions may be only its network definition.** The directives that
   define the model rather than a protocol are allowed: `generate_network`, and `setOption`
   with the siblings the scanner has always kept in the model text (`setModelName`,
   `substanceUnits`, `version`). Comments and blank lines are fine. Anything else is a
   protocol action and is refused: `simulate*`, `parameter_scan`, `bifurcate`,
   `setParameter`, `setConcentration`, `addConcentration`, the save and reset actions,
   `write*`, `visualize`, `readFile`, and any action PyBNF does not know. What "an action" is
   comes from the fitter's own scanner, so the rule sees exactly the lines the fit would run,
   in the shapes BNG2.pl accepts: indented, commented, continued with a backslash, loose after
   `end model`, or numbered.
2. **One check, three places.** `BNGLModel.require_no_protocol_actions` raises a `PybnfError`
   naming the model file and each offending line (its first physical line and the statement),
   says why, and tells the user to move a `setParameter` or `setConcentration` into a
   `condition:` (or into the model's parameters or seed species) and to delete leftover
   `simulate`, `write*` and similar lines. It runs:
   - **at config load**, for every BNGL model that `experiment:` lines simulate, before any
     experiment is synthesized onto it;
   - **at PEtab export**, for every BNGL model the export writes, with the same message, so
     the export never drops an action the fit ran (the #900 whitelist is gone);
   - **at PEtab import**, for every BNGL model in the problem, with a message that says the
     PEtab tables define the protocol, before any file is written.
3. **Edition 1 is unchanged.** There the actions block is the protocol. So is an edition-2
   model that no `experiment:` line simulates, one still bound the legacy way
   (`model = b.bngl : b.exp`): its actions are its protocol and it keeps them. A model bound
   both ways is refused, and the message says its simulations belong to the legacy binding,
   so the user moves that data to `experiment:` lines rather than deleting a line the binding
   needs.
4. **A numbered action line is read as BNG2.pl reads it** (#963 item 3). BNG2.pl removes a
   leading line index from an action (`BNGModel.pm`, `s/^\d+\s+//`). The scanner now does
   too, for every use it makes of the line: recognising `generate_network` and `setOption`,
   reading a simulation's suffix, the action list the bngsim classifier and both bridges read,
   and the protocol block. A numbered `setOption`, which stays in the model text outside any
   block, is written without its index, because BNG2.pl strips the index only inside a block
   and would otherwise skip the line.

## Why a rule and not a defined splice

The alternative was to give the splice a meaning: decide, action by action, what a
hand-written action ahead of the experiments does to each of them, and make both backends and
the PEtab export agree. The cases above show how many decisions that is, and each needs a
backend fix, a test and an export rule. None of the 807 edition-2 confs in `examples/` (115)
and BNGL-Models' `pybnf-jobs` (692) uses a model with a protocol action, so nothing that
exists needs the splice. The conf already expresses everything such an action was used for:
a `condition:` sets parameters and species, `preequilibrate:` equilibrates, and the
`generate_network` conf key caps a network. With the rule, an edition-2 model's action list is
the network definition followed by the synthesized experiments, which is what ADR-0151 and
the PEtab export already assumed.

## Checked, path by path

- **BNG2.pl** (`BNGLModel`, `NetModel`): the fit's model text is the model, its
  `generate_network` line, and the synthesized experiments. A numbered `generate_network` is
  now the network definition; before, it was kept as an action and run against the `.net`
  file PyBNF had generated, and BNG2.pl aborted every simulation.
- **bngsim `.net`** and **bngsim network-free**: both read `model.actions`, which for an
  edition-2 model now holds only synthesized lines. A numbered edition-1 action reaches the
  classifier without its index, so it routes to the bridge rather than falling back to
  BNG2.pl (or being refused under `bngl_backend = bngsim`).
- **SBML and Antimony models** have no BNGL actions and are not affected.
- **Multi-model jobs**: each BNGL model is checked on its own; a legacy-bound model in the
  same job keeps its actions (decision 3).
- **Multiple shooting, gradient fits, `check`**: all load through the same Configuration, so
  the rule applies to them unchanged.
- **The `generate_network` conf key** (#473, #901): unchanged. A model's own
  `generate_network` line still takes precedence over the key, as documented; the export
  writes whichever the fit used.
- **Emit-set pruning** (ADR-0069): a BNGL model that experiments simulate can no longer mix in
  hand-written simulations, so its separability check now fails only for a legacy
  `time_course` / `param_scan` key on the same model.

## Consequences

- A job that ran before can now stop at load, when its edition-2 model carries a protocol
  action. The message names every such line, up to ten.
- #941 (a hand-written `parameter_scan` restores its parameter on bngsim but not on BNG2.pl)
  can no longer reach an edition-2 job; it remains an edition-1 difference.
- The exporter's `_require_droppable_actions` and its action lists are deleted. The export
  tests that asserted a leftover `simulate` exports fine now assert that it is refused, at
  load and at export, with one message.

## Alternatives rejected

- **Keep the #900 whitelist and extend it.** It made the export safe but left the fit running
  whatever the model held, and it needed a new argument for every action, method and backend.
- **Warn and continue.** The result would still depend on the splice.
- **Translate a hand-written `setParameter` into a condition automatically.** It would
  reproduce one case of the splice, and a model file that means something different under the
  fitter than under BNG2.pl would stay.
- **Hold legacy-bound models in edition 2 to the rule too.** Their actions are their only
  protocol; refusing them would break a supported surface for no gain.
