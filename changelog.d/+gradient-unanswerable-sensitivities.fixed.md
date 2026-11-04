- **A gradient fit whose sensitivities cannot be computed is refused at job start instead
  of finishing at `inf`.** On a model whose analytic sensitivity right-hand side bngsim
  declines, a fitted species initial value needs a column CVODES' difference quotient
  cannot supply, and so does every state `job_type = ms` carries across a knot. PyBNF scored
  that as a failed simulation at every point, so the fit ran every job to failure and
  reported "Stop criterion satisfied with objective function value of inf" with exit status
  0; on bngsim 0.17.0 the start-of-job check also said nothing. Under
  `sensitivity_fallback = warn` (the default) and `error`, the fit is now refused at job
  start, naming the model and bngsim's reason.
