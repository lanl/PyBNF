- **A gradient fit refuses a scored parameter scan over a parameter it differentiates.** The
  scan writes its own value to the swept parameter at every dose, so the loss does not move
  with the fitted value, but the gradient column for it is the derivative at the dose. For a
  fitted `kr`, the assembled gradient was -472.32 where a central difference gives 0. Such a
  scan, `.net` or SBML, over a fitted parameter or over one a condition binds a fitted
  parameter to, is now refused at load, naming the scan and the parameter.
