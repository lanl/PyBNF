- **With numpy 1.26 on an AVX-512 CPU, a run's results could depend on memory addresses, not only on its seed.** Converting a strided column of values (`10.0 ** u[:, j]`, `log10`) gives a last bit that numpy picks by memory address. It runs Intel's SVML or numpy's own AVX-512 code, or the C library, and the two differ for about a quarter of the values.
  - The Liu–West filter (`job_type = lwf`) did not always write the same files twice. An `lwf_continue` run did not always write what the uninterrupted run writes.
  - `am` did not always draw the same chain from the same `random_seed`, through the history it seeds its adaptive covariance from.
  - The sampler histograms and `from_pybnf`'s sampling-space posterior could differ in the last bit.

  Each conversion now takes a contiguous copy. numpy 2.5.4 on the same CPUs gave the same bits either way.
