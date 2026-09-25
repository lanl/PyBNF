- **A bootstrap replicate of a multiple-shooting fit (`job_type = ms`) no longer reports a
  start belonging to the replicate before it.** Starts accumulated across replicates, so a
  later replicate could report a start fitted to different resampled data as the one behind
  `Results/continuity_defects.txt` and the best stage trace.
