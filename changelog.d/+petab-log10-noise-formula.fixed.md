- **Converting a PEtab v1 problem with petab 0.10.0 no longer inflates a log10 observable's
  sigma by ln 10.** petab 0.10.0 writes a v1 `log10` observable's noise formula as
  `log(10) * (<formula>)`, which is right for the natural-log noise family petab writes but
  wrong under the `observableTransformation = log10` column that
  `petab1to2_preserve_scale` restores. A log10 observable with a numeric sigma then imported
  with a sigma 2.3 times the stated one, with no error. The converter now puts the v1 noise
  formula back, and refuses any other rewrite of it, so the noise formula is the same under
  petab 0.9 and 0.10.
