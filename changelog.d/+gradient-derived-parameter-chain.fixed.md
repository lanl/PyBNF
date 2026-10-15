- **A gradient fit no longer counts the chain through an SBML parameter `initialAssignment`
  twice.** bngsim keeps such a parameter (`beta = R0*g/N`) tracking its inputs, so their own
  sensitivity columns already carry it; PyBNF added the derived parameter's column on top.
  That was harmless only while bngsim reported that column as zero, and on bngsim builds
  with lanl/bngsim#941 the gradient came back 2x to 4x too large. The router now asks the
  model which derived parameters it keeps live, the derived value is no longer written back
  where bngsim already holds it (a write one ulp off had overridden the expression), and a
  run model that disagrees with its experiment's routing stops the fit with the parameter
  named. A write to a parameter or species the model's own `initialAssignment` defines (a
  free parameter, a condition, a scan or an equilibration condition) is refused on the
  gradient path when the evaluation also changes something an `initialAssignment` reads,
  since the assignment then overwrites the write.
