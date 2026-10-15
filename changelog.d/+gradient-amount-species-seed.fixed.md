- **A gradient fit no longer counts twice the seed of an amount species in a compartment whose
  size is not 1, when the parameter that seeds it is also read by a rate law.** The SBML and
  Antimony backends reported such a species to the router as not seeded on the parameter's
  own sensitivity column, on a measurement from before both columns were read into PyBNF's
  units and bngsim divided that seed by the volume. The router then added the
  initial-condition term on top of a column that already carried it; for `S = 100*k` with
  `k` a rate constant the gradient came back about 2x on bngsim 0.16.0. The backend now
  reports what bngsim seeds.
