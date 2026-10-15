- **A gradient fit keeps the dependence of a species' initial value on the parameters it is
  defined from.** PyBNF saves an initial state as the reset target in several places: every
  `.net` evaluation at its start, each dose of a scan, each `saveConcentrations()` in the
  actions, and an SBML/Antimony evaluation that writes a species. On bngsim builds with
  lanl/bngsim#982 a saved state carries no parameter seeding, so a free parameter such as
  `S0` in `S() S0`, or one an `initialAssignment` reads, got a gradient column of exactly
  zero for its initial-value term. PyBNF now reads the seeding before each such save and
  declares it again after every reset back to it, a multiple-shooting knot is declared as
  depending on no parameter (bngsim releases before #982 had seeded it), and a run whose
  seeding differs from the routing's stops with the species and parameter named. A scored
  `ss_method=>"newton"` scan whose fitted parameter seeds a conserved species is refused on
  a bngsim build whose Newton solve drops that term (lanl/bngsim#704).
