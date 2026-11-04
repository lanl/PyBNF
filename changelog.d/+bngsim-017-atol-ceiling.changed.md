- **On bngsim 0.17.0, a number given as `sbml_atol`, the scalar PyBNF derives for an SBML
  model, and a BNGL action's `atol=>` are per-species ceilings.** bngsim 0.17.0 reads a
  scalar `atol` as `min(atol, rtol*max(|x_i(0)|, floor))` for each species
  (lanl/bngsim#1420), and PyBNF hands the number over as written, so no species is held
  looser than it and one far below it is still resolved. Through bngsim 0.16.0 the number
  held every species exactly. The tolerance docs now say so. The sensitivity-fallback
  tests decline on `floor()`, since 0.17.0 differentiates `abs()` of the state
  (lanl/bngsim#1126).
