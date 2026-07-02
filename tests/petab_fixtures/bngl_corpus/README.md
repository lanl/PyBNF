# Curated BNGL corpus fixtures

Public community BNGL models, vendored from the `bng_parity` corpus
(`PyBNF-Private/.../bng_parity`, itself fetched from pinned public repos:
RuleWorld/RuleHub, wshlavacek/BNGL-Models, richardposner/RuleMonkey).
Each is the input to `tests/test_petab_bngl_corpus.py`, which asserts our
BNGL reader enumerates the same entities BNG2.pl does. Chosen for feature
coverage, not at random.

| model | source | upstream path | exercises |
|---|---|---|---|
| `An_2009.bngl` | rulehub | `Published/An2009/An_2009.bngl` | labeled seed species; large rule-based (TLR signaling) |
| `Barua_2009__PATCHED.bngl` | rulehub | `Published/Barua2009/Barua_2009.bngl` | uses the `begin species` block alias |
| `Chattaraj_2021.bngl` | rulehub | `Published/Chattaraj2021/Chattaraj_2021.bngl` | indexed seed species |
| `LR.bngl` | rulehub | `Tutorials/NativeTutorials/LR/LR.bngl` | uses the `begin species` block alias |
| `LRR.bngl` | rulehub | `Tutorials/NativeTutorials/LRR/LRR.bngl` | uses the `begin species` block alias |
| `Motivating_example_cBNGL.bngl` | rulehub | `Tutorials/MotivatingexamplecBNGL/Motivating_example_cBNGL.bngl` | compartmental BNGL |
| `Ras_bistability_v2.bngl` | bngl_models | `my_models/ode/Ras_bistability_v2.bngl` | component reordering vs BNG canonical order |
| `Rule_based_egfr_tutorial.bngl` | rulehub | `Published/Rulebasedegfrtutorial/Rule_based_egfr_tutorial.bngl` | indexed seed species |
| `akt-signaling.bngl` | rulehub | `Examples/biology/aktsignaling/akt-signaling.bngl` | states + bonds; multisite phosphorylation |
| `apoptosis-cascade.bngl` | rulehub | `Examples/biology/apoptosiscascade/apoptosis-cascade.bngl` | states + bonds; compartment-free signaling |
| `bcr-signaling.bngl` | rulehub | `Examples/biology/bcrsignaling/bcr-signaling.bngl` | $ clamp; states + bonds (B-cell receptor) |
| `blood-coagulation-thrombin.bngl` | rulehub | `Examples/biology/bloodcoagulationthrombin/blood-coagulation-thrombin.bngl` | large rule-based; local functions |
| `bmp-signaling.bngl` | rulehub | `Examples/biology/bmpsignaling/bmp-signaling.bngl` | complex formation; $ clamp |
| `brusselator-oscillator.bngl` | rulehub | `Examples/biology/brusselatoroscillator/brusselator-oscillator.bngl` | small sanity model |
| `catalysis.bngl` | bngl_models | `my_models/ode/catalysis.bngl` | compartmental BNGL; $ clamp under @compartment prefix |
| `egg.bngl` | rulehub | `Published/Hlavacek2018Egg/egg.bngl` | line continuations; bare-molecule seed (t -> t()) |
| `elephant_EFA.bngl` | rulehub | `Published/Hlavacek2018Elephant/elephant_EFA.bngl` | bare-molecule seed; continuations |
| `energy_transport_pump.bngl` | rulehub | `Examples/energy/energytransportpump/energy_transport_pump.bngl` | energy patterns |
| `example1.bngl` | rulehub | `Tutorials/example1/example1.bngl` | indexed parameters + indexed seed species |
| `genetic_bistability_energy.bngl` | rulehub | `Examples/genetics/geneticbistabilityenergy/genetic_bistability_energy.bngl` | energy patterns |
| `immob_equiv_lig_sites.bngl` | bngl_models | `my_models/nf/immob_equiv_lig_sites.bngl` | line continuations in functions; local functions |
