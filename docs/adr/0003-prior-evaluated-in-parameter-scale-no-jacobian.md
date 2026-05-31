# Prior and proposal share one scale; the prior is evaluated in that scale (no Jacobian)

PyBNF free parameters of a log type operate entirely in `u = log10(θ)`: the
proposal arithmetic (`FreeParameter.add`), the bounded-domain reflection, the
R-hat chain history, and the prior density all live in that space. The
`PRIOR-JAC` punchlist investigation empirically confirmed this is a deliberate,
self-consistent parameterization in which the posterior target is defined
directly over `u` — so there is **no** change-of-variables Jacobian, and adding
one would be a bug. We decided the extracted `Prior` abstraction keeps **scale as
a parameter-level property shared with the proposal arithmetic**, and evaluates
the prior density in the parameter's own scale. We do **not** support a prior
specified on a different scale than the parameter (PEtab's independent
`parameterScale` prior), because that reintroduces the Jacobian and a correctness
surface, and no current use-case needs it.

## Considered Options

- **PEtab-style independent prior-scale with an explicit change-of-variables Jacobian.** Deferred, not rejected forever: revisit only when a concrete use-case requires a prior on a scale other than the parameter's own.
