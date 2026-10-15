- **PyBNF now requires bngsim 0.16.0 or later** (`bngsim>=0.16.0,<1`, and the same for the
  `antimony` extra and the CI install). The gradient routing and its run-time checks are
  written against the answers bngsim 0.16.0 and later give about derived parameters and
  initial-condition seeding; 0.15.x is no longer supported. A gradient-based fit also checks
  the installed version when it starts, so a 0.15.x build installed from source is refused,
  naming its version and the minimum. Gradient-free fits keep the old minimum.
