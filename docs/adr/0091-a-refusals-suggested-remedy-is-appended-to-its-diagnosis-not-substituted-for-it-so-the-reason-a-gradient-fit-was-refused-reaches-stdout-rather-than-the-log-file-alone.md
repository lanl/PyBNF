# A refusal's suggested remedy is appended to its diagnosis, not substituted for it, so the reason a gradient fit was refused reaches stdout rather than the log file alone (issue #527)

**Status: Accepted and implemented (2026-08-01).** `PybnfError` gains a third argument, `hint`,
which **appends** to the user-facing message. Every gradient-path refusal — and every
`profile_likelihood` refusal, which had the same shape — now prints its own diagnosis followed by
its remedy; `user_message`'s **replace** semantics are unchanged and stay available for the case
they were written for. No message reaches the user with less in it than before.

## Problem

`PybnfError` carries two texts — one for the log, one for the user — and the second replaces the
first:

```python
def __init__(self, log_message, user_message=None):
    self.log_message = log_message
    self.message = user_message if user_message else log_message
```

A raise site therefore has to choose. A site whose diagnosis is *too technical to print* uses the
slot correctly: the log gets `Column Stot_SD not found`, the user gets the same fact spelled out.
But a site whose remedy is *generic* spends the slot on the remedy, and its diagnosis — the only
part that differs between one refusal and the next — never reaches stdout.

Every gradient-path gate was shaped that way. Four unrelated conditions refuse a gradient fit — a
legacy (edition < 2) config, a backend without forward sensitivities, a discrete-event model, an
objective the assembly cannot differentiate — and all four passed their diagnosis as
`log_message` and one shared `_FALLBACK_HINT` as `user_message`. So all four printed one sentence:

```console
$ pybnf -c Smith_BMCSystBiol2013.conf -o
Error: Gradient-based fitting is not available for this fit; use a metaheuristic job_type
instead (e.g. job_type = de, the default, or pso / ss / cmaes), which needs no gradient.
```

That sentence names no cause, reads as *this problem is not eligible* rather than *this specific
feature is unsupported*, and does not hint that a reason exists elsewhere. `-L debug` does not
change it. The reason was in `bnf_<timestamp>.log` and nowhere else:

```text
ERROR MainProcess Gradient-based fitting (job_type = gntr) requires smooth, differentiable
dynamics, but model 'model_Smith_BMCSystBiol2013' contains discrete events (a state-dependent
discrete jump). Forward output sensitivities go stale across such a jump, so bngsim cannot
supply the gradient there.
```

The cost is concentrated in triage. Told the two sentences apart, a user knows whether they are
looking at a two-line config fix (`edition = 2`) or a known upstream gap (event-aware
sensitivities). Told only the fallback, they cannot tell, and the natural reading — *this fit
cannot be differentiated, full stop* — is wrong for three of the four cases. The inconsistency
compounded it: refusals that pass no `user_message` at all, such as the `output_sensitivities`
capability gate, printed their full reason, so the surface gave no way to know that a bare hint
means a reason is being withheld.

## Decision

### `hint` is a third slot, and it appends

```python
def __init__(self, log_message, user_message=None, hint=None):
    self.log_message = log_message
    self.hints = [hint] if isinstance(hint, str) else list(hint or ())
    self.message = user_message if user_message else log_message
    for h in self.hints:
        self.message += '\n  -> ' + h
```

The two knobs are now orthogonal and each does one job:

| slot | relation to `log_message` | for |
|---|---|---|
| `user_message` | replaces | the same fact, restated at a different depth |
| `hint` | appends | a *different* fact — what to do about it |

A hint is rendered as an indented `->` line, one per hint, so a refusal with more than one way out
lists them rather than running them together. `hints` is a plain list of strings; passing a bare
string is the common case and is normalized to a one-element list.

The result on the reproducer:

```console
$ pybnf -c Smith_BMCSystBiol2013.conf -o
Error: Gradient-based fitting (job_type = gntr) requires smooth, differentiable dynamics, but
model 'model_Smith_BMCSystBiol2013' contains discrete events (a state-dependent discrete jump).
Forward output sensitivities go stale across such a jump, so bngsim cannot supply the gradient
there.
  -> Use a metaheuristic job_type instead (e.g. job_type = de, the default, or pso / ss /
     cmaes), which needs no gradient.
```

### The log keeps the diagnosis alone

`hint` changes only `message`. `pybnf.pybnf` still logs `e.log_message`, which is unchanged at
every site, so the log file reads exactly as it did. At every migrated site — none of which pass a
`user_message` — stdout is now a superset of the log line, the relation the gradient tests assert
directly (`e.log_message in e.message`).

### The gradient path and `profile_likelihood` migrate

`_FALLBACK_HINT` is reworded from a standalone sentence ("Gradient-based fitting is not available
for this fit; use a metaheuristic…") to a suffix ("Use a metaheuristic…"), since the diagnosis it
now follows has already said what is unavailable. Seven gradient-path sites move to `hint=`: the four gates and
the per-evaluation objective refusal in `gradient_base.py`, `gntr`'s EFIM-Hessian refusal and
`trf`'s exact-least-squares refusal (both of which pointed at `lbfgs` while dropping the reason
`lbfgs` was needed), and `routing._resolve_condition`'s unknown-condition refusal.

Two gates gained a second, more specific hint above the fallback, because for them the fallback is
the *worse* of two remedies — the fit is fine, the surface or the backend is wrong:

* the edition gate suggests `edition = 2` and the new-era surface first;
* the backend gate suggests simulating through bngsim (`sbml_backend = bngsim` for an SBML model),
  which supplies the forward sensitivities the model's current backend does not.

`profile_likelihood`'s three refusals were in the same shape and migrate with them, because the
fact each one dropped is one the user cannot reconstruct from their own config:

* `profile_likelihood_params` naming an undeclared parameter printed *"List only free-parameter
  ids to profile (or omit the key to profile all of: k, S0, …)"* — the valid ids, never the
  rejected one;
* the bounded-box gate printed *"Declare each parameter with a bounded prior"* — never which
  parameter was unbounded;
* the non-integrable reference point split its diagnosis across both slots, so *"could not
  simulate its reference point (the box center)"* went to the log and *"The point is
  non-integrable at these parameters"* to the user. That sentence moves into `log_message`, where
  it joins the rest of the diagnosis, leaving the two remedies as the hint.

The ~90 other two-argument raise sites are **untouched**. They were surveyed, and they use the
slot as designed: `config.py`, `parse.py`, `objective.py`, the samplers, and the noise sources put
a terse diagnosis in the log and a fuller retelling of the *same* fact in `user_message` —
`"Invalid proposal 'x'"` / `"Config key 'proposal' must be 'de', 'whitened', or 'kalman'."`.
Rewriting those would churn working messages. Any site that later finds itself carrying a remedy
in `user_message` moves to `hint` when touched; nothing forces a sweep.

### `args` is left alone

`PybnfError` still does not call `super().__init__`, so `str(e)` remains whatever `BaseException`
built from the positional arguments. Passing `hint` as a keyword keeps it out of `args`, which
means `pytest.raises(..., match=…)` now matches against the diagnosis rather than a
`('log', 'user')` tuple repr — a small improvement, and the reason the two tests that read hint
text out of `str(e)` now assert on `.message` explicitly, which is the thing under test anyway.
Making
`str(e)` equal `message` would be the tidier end state, but it would silently redirect every
existing `match=` on a two-argument site from the log text to the user text, and that is a
separate change from this one.

## Consequences

- The four gradient refusals are pairwise distinct on stdout and each names its own cause. A user
  triaging an imported collection can tell a config fix from an upstream gap without opening a log
  file. A `profile_likelihood` refusal names the parameter at fault.
- Every message is a superset of what it printed before; no user-facing text was removed.
- Log files are unchanged except at the non-integrable reference point, whose log line gains the
  "the point is non-integrable at these parameters" clause that used to reach only the user.
- `user_message` keeps its meaning, so nothing outside the two migrated modules changed behavior.
- A raise site now has to choose deliberately: a remedy belongs in `hint`, a retelling in
  `user_message`. The class docstring states the distinction, which is what made the original
  misuse easy — there was one slot and two things to put in it.
