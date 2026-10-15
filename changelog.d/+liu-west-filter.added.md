- **`job_type = lwf`, the Liu–West filter: forecast count data that arrive one interval at a
  time.** At each row it moves the particles' free parameters by the Liu and West (2001) kernel,
  weighs them by the job's own cumulative `neg_bin` line on the interval's increment, and
  resamples. Its output, under `Results/LWF/`, is a forecasting sample of a model whose free
  parameters drift, not a posterior. `lwf_continue = 1` assimilates only new rows from a state
  file. Every key it does not read is refused by name.
