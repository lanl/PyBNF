- **A `setConcentration` amount written over a parameter is now differentiated at the values
  the parameter has when it runs.** PyBNF replaced every parameter in the amount by its value
  in the `.net` file, constants included. So `Atot*dose` after `setParameter("dose",3)` gave
  `Atot` a gradient column a third of the true one, and the same amount before a scan over
  `dose` used the file's `dose` at every point. On bngsim 0.16.0, an amount such as
  `Atot*(dose-1)` that is zero at the file's value also let a Newton scan run that should
  have been refused. Only derived parameters that still follow their expression are
  expanded now, and the row is evaluated at the run's own values: the scanned dose, an
  earlier `setParameter`, the fitted values. A seed that is zero at one dose is kept, so the
  steady-state refusal does not depend on it. An amount reading a parameter that a condition
  sets to a constant is no longer refused as reading a fitted parameter.
