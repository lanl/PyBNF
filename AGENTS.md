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
