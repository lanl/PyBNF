- **A `.net` gradient fit asks each condition's runs for only the sensitivity columns that
  condition's experiments read.** The request used to be the union over every experiment, so
  a condition that binds `kf` to a fitted parameter made every other condition differentiate
  `kf` too. bngsim cannot scan a parameter it is differentiating, so a carried (no reset)
  dose-response scan of `kf` in the wildtype was refused, though nothing there reads that
  column. It now runs, and its gradient matches central differences. Each run also computes
  fewer sensitivity columns.
