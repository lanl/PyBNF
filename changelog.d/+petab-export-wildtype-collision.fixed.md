- **PEtab export refuses a job condition named `wildtype` wherever it would collide with the
  base condition the exporter writes as `cond_wildtype` (#906).** When a fit parameter is also
  perturbed by a condition, the exporter writes its own `cond_wildtype` to re-pin it, and a job
  condition of that name was written under the same id: a wash-out's or pre-equilibration's
  period then silently took the job condition's targets, or lost them. Rename the condition.
