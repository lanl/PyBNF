- **A gradient fit whose sensitivities cannot be computed stops with an error instead of
  finishing at `inf`.** On a model whose analytic sensitivity right-hand side bngsim
  declines, a fitted species initial value needs a column CVODES' difference quotient
  cannot supply, and bngsim refuses some declined models outright. PyBNF scored each of
  these as a failed simulation at every point, so the fit ran every job to failure and
  reported "Stop criterion satisfied with objective function value of inf" with exit status
  0. On bngsim 0.17.0 the start-of-job check also said nothing. The fit is now refused at job
  start, naming the model and bngsim's reason, or at the first simulation under
  `sensitivity_fallback = ignore`.
