- **A gradient fit that overrides a derived parameter is refused where bngsim would
  differentiate through the expression the override replaced.** When a fitted value, a
  condition, a `setParameter` action, a scan or a pre-equilibration measurement condition
  writes a number to `k2 = 2*k1`, `k1` no longer moves `k2`. bngsim 0.16.0 still puts the
  chain through `k2` into `k1`'s column: about 20x too large when `k2` was fitted beside `k1`.
  A one-model check run once per process now decides whether the installed bngsim does this.
  Where it does, such a run on the `.net`, SBML and Antimony backends and in multiple
  shooting stops and names the parameter. A relative condition such as `Btot * 2` on a
  derived parameter whose inputs are fitted is refused at load on every build. Its value
  moves with those inputs, but it is written as a number, so no gradient column carries that
  dependence.
