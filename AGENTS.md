# Instructions for coding agents

These rules apply to any automated coding assistant working in this repository. They
supplement [CONTRIBUTING.md](CONTRIBUTING.md), which applies to everyone.

## Fail loudly; never fall back to a silently wrong result

PyBNF's output is a scientific result. A run that stops with a clear error costs the user a
few minutes. A run that completes with a wrong objective value, a wrong best fit, or a wrong
posterior can end up in a paper. So when code meets input it cannot handle correctly, it must
stop with an error that names the cause. It must never substitute a default and carry on.

This is the most common defect class in the codebase. A whole-codebase audit on 2026-09-23
confirmed it repeatedly (#817–#834).

- **Resolve every name when the configuration loads.** An observable, data column,
  condition or mutant target, `noise_model` observable, suffix, or parameter that the
  configuration names must match something in the model or the data. If it does not, refuse
  the configuration and say which name was unmatched and what it could have matched. Never
  take the intersection of two name sets and score whatever overlaps (#817, #818).
- **Refuse combinations that are not supported.** When a new option does not compose with an
  existing one, refuse the combination at load time and say so. Partial support, where one
  code path honours the option and another ignores it, is the worst outcome (#820, #822,
  #825).
- **A configuration or data error is not a simulation failure.** Do not catch an error that
  reflects a problem with the configuration or the data and turn it into a failed simulation,
  a penalty score, or a line in the log file. Let it end the run. Only a genuine simulation
  failure, such as an integrator that cannot proceed, is scored as a failed simulation (#817).
- **No silent default for missing input.** `dict.get(key, default)`,
  `getattr(obj, name, default)` and `except ...: pass` are acceptable only when the default
  is the documented, correct meaning of absence. If absence means the user made a mistake,
  raise.
- **A warning is not a substitute for an error.** If the result will be wrong, raise; do not
  warn and continue. Errors reach the console, not only the log file.
- **Every new refusal gets a test** showing that the bad input is rejected with an error
  naming the cause.

If you find an existing silent fallback while working on something else, do not extend the
pattern. Report it so it can be filed.

## Done means checked, not green

A passing test suite shows that each feature does what its author intended, for the inputs
its author thought of. The 2026-09-23 audit found its bugs in the other cases: two features
used together, one code path updated and its siblings left behind, and input nobody expected.
A change is done only when all of the following hold, and the pull request says how each one
was met:

- **An independent oracle.** At least one test checks the result against a calculation that
  does not go through the code under test: numpy by hand, an analytic solution, a published
  value, or a different backend. A snapshot of the code's own earlier output only fixes the
  current behaviour in place; it does not show that the behaviour is right.
- **Every sibling path.** Much of PyBNF does one job in several places. Examples are the
  simulation backends (BNG2.pl, bngsim with `.net` models, bngsim network-free, RoadRunner,
  bngsim SBML/Antimony), the objective's value and its gradient, the fit and multiple shooting
  (`job_type = ms`), and a fit and its PEtab export and import. List the siblings of what
  you changed. For each, say whether it was changed, why it does not apply, or that it now
  refuses the case at load. The `postprocess` key is the cautionary example: the fit applies
  it, but the gradient, multiple shooting, and PEtab export all ignored it.
- **Every interaction of a new option.** List the existing options the new one can meet. For
  example: normalization, `cumulative`, `time_error`, `noise_model`, `bootstrap`,
  `smoothing`, `postprocess`, conditions and mutants, pre-equilibration, gradient fitting,
  and each `fit_type` and `job_type` it applies to. For each pair, either test it or refuse
  it at load (see above). Leaving a pair unconsidered is not an option.

## The author does not verify their own work

Tests written in the same session as the code check that session's understanding of the
code, and they share its blind spots. So:

- **Before a pull request is ready, a reviewer that did not write the change must examine it.**
  That can be a fresh session or a separate subagent, but not the author's own context. Its
  brief is to find ways the change produces a silently wrong result and to reproduce each by
  running the code, not only by reading it.
- **The reviewer adds at least one test the author did not write.**
- **Do not describe your own change as verified.** Say what you tested. Say plainly that
  independent review is still to come, or say what it found.
