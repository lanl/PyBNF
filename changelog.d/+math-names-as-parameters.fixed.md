- **A bngsim model parameter named `e`, `pi`, `log`, `atan2`, `ceil`, `floor` or `pow` is now
  read as the parameter, as BNG2.pl and bngsim read it, not as the math name**, in a species
  initializer of a `.net` model, a `setConcentration` expression and a network-free model's
  parameter block, where a fit could start from a wrong initial condition or derived parameter.
