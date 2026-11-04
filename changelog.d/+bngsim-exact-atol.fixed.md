- **On bngsim 0.17.0, an absolute tolerance PyBNF passes is used as written again.**
  bngsim 0.17.0 reads a bare number `atol` as a per-species ceiling, taken from the state
  each run starts from, so a stated `sbml_atol`, the tolerance PyBNF derives for an SBML
  model, and a BNGL action's `atol=>` stopped meaning that number. On a model whose species
  sit near `1e-8`, `sbml_atol = 1e-8` and `1e-10` gave the same run. PyBNF now passes each
  one as `bngsim.ExactAtol`, on the simulate, segment and multiple-shooting paths, so it
  integrates as it does on bngsim 0.16.0 and under BNG2.pl. A BNGL action that gives no
  `atol=>` still takes bngsim's own default.
