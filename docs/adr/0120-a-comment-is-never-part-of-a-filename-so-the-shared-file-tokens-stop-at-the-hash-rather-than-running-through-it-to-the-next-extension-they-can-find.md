# A comment is never part of a filename, so the shared file tokens stop at the `#` rather than running through it to the next extension they can find (issue #599)

**Status: Accepted and implemented (2026-08-20).** Closes the half of the hazard that
ADR-0116 (#586, PR #596) deliberately left open, and does it at the token rather than at one
declaration's guard.

`parse.py`'s three file tokens were unanchored, lazy regexes:

```python
model_file = pp.Regex(r".*?\.(bngl|xml|ant|target)")
exp_file   = pp.Regex(r".*?\.(exp|con|prop)")
param_file = pp.Regex(r".*?\.tsv")
```

Both properties are deliberate and both are kept. **Unanchored** is what lets a path start
with anything a filesystem allows — `../`, `~`, `/abs`, a drive letter. **Lazy** is what
makes a comma list stop at the first extension instead of swallowing the rest of the line.
Neither is the defect.

## The defect

Nothing stopped either property from operating *across a comment*. A `.*?` will start
mid-line and run through spaces and a `#` to reach the next extension it can find, so a
stray trailing comma followed by a comment that happens to mention a file became a second
**file**:

```
model: a.xml, # note about b.xml            -> models   = {'a.xml', '# note about b.xml'}
model = a.xml : d.exp, # note about e.exp   -> exp_data = {'d.exp', '# note about e.exp'}
```

Both spellings of a model-data mapping were affected — the new-era `model:` declaration and
the legacy `model = … : …` form — and the second invents an **exp** file, not just a model.
Measuring the reach found three more consumers with the identical defect, which the issue
had not named: `mutant = … : …`, `experiment: … data:`, and the `data =` key all invent a
phantom `.exp` the same way.

### The failure is loud, but it is loud about the wrong thing

The issue filed this as non-urgent because the bogus entry dies in the loaders rather than
being integrated. That is true, and it is the reason this was a defect worth measuring
rather than an emergency. What measuring found is that "loud" is doing less work than it
sounds: the message the user actually gets depends on which extension their *comment*
happened to contain, and none of the five point at the comma.

| invented entry | what the user is told |
|---|---|
| `# … b.bngl` | `Model file # note about b.bngl was not found.` |
| `# … b.xml` | `Failed to load model # … b.xml - There were errors in parsing this SBML file.` |
| `# … b.ant` | `Antimony model support was requested, but requires optional dependency 'antimony'.` |
| `# … e.exp` | `Action not specified for '# … e.exp'` — *"your model file needs `suffix=>…`"* |
| `# … e.con` | `Constraint file # … e.con was not found` |

Only the first is the "confusing message about a file they never wrote" the issue predicted.
The `.xml` row — which is the form the issue's own repro uses — reports a **parse error in
the user's SBML**, about a file that does not exist. The `.ant` row tells them to install an
optional dependency they never asked for. The `.exp` row is reached from `_check_actions`
*before* `_load_exp_data` can say "not found", so it sends them into their BNGL to add a
`suffix=>` action for a measurement that is a sentence in English.

This also answers the open question the issue attached to it — whether any `.exp`/`.con`/
`.prop` consumer would *silently* accept a bogus entry rather than erroring. **None does.**
Every path errors. The data side is simply the side that misdiagnoses hardest.

### A second failure the fix also closes

Measuring the blast radius turned up a second failure of the same regex that nobody had
filed, on a different mechanism:

```
model = a.xml : none # note about e.exp     -> exp_data = {'none # note about e.exp'}
```

The alternation is `(_DelimitedList(exp_file) ^ nonetoken)`, and `^` is pyparsing's `Or` —
**longest match wins**. `nonetoken` matches 4 characters; `exp_file`, free to cross the `#`,
matched all 23 of `none # note about e.exp`. So a trailing comment on a `: none` line
converted *"this model has no data"* into *"this model has one data file"*. It fails
downstream like the others, but the declaration it corrupts is the one that says a model is
deliberately unmeasured, which is the last place a user would look.

## The decision

**A `#` is never part of a filename, because `#` is what ends the part of a line a filename
could live in.** All three tokens take a `[^#\n]` character class:

```python
model_file = pp.Regex(r"[^#\n]*?\.(bngl|xml|ant|target)")
exp_file   = pp.Regex(r"[^#\n]*?\.(exp|con|prop)")
param_file = pp.Regex(r"[^#\n]*?\.tsv")
```

`[^#\n]` rather than `[^#]` keeps the historical no-crossing-a-newline behaviour of `.`
explicit now that the class is spelled out. Unanchored and lazy are untouched.

**It lands on all three at once, and that is the whole reason it is a token change rather
than a second guard.** ADR-0116 closed the *new* half of this hazard — a comma followed by a
tolerance field — with a negative lookahead scoped to the `model:` declaration's file token,
and said why it went no further: the legacy `model = … : …` form still accepted `#`, and two
spellings of one declaration disagreeing about which files exist would be worse than the
corner it would close. That reasoning is still right, which is why the narrowing could only
ever be applied to the shared tokens, where both spellings and every other consumer —
`mutant`, `data =`, `condition: model:`, and `experiment:`'s `model:` / `data:` /
`measurement_params:` fields — get it simultaneously.

(The issue also listed `normalization` and the PEtab import lines as running through these
tokens. They do not: `normalization` takes its own permissive `anything` word and is
resolved by separate code, and PEtab import does not go through `parse.py` at all. Both are
unaffected, and the 41-shape sweep below confirms it.)

The `model_decl_file` lookahead from ADR-0116 **stays**. The token now covers the comment
half of what that guard was built for, but the guard still covers the rest of the line:
`model: a.xml, atol: 1e-4, b.xml` has no comment in it and still needs refusing.

### The one thing that narrows

A filename containing a literal `#` is no longer expressible. This is a real reduction in
what the grammar accepts and it is accepted deliberately:

* Nothing in the repository, the tutorial suite, or the `BNGL-Models` corpus has such a
  filename — checked, zero hits.
* A `.conf` format that supports `#` comments could not round-trip such a name anyway; the
  old behaviour did not so much *support* `#` filenames as fail to notice them.
* The two spellings must agree either way, so this was never a per-form choice.

## The measured blast radius

41 line shapes were run through `ploop` before and after — every construct that reaches one
of the three tokens, plus the ADR-0116 tolerance fields, plus relative/absolute/`~`/spaced
paths. **9 changed; the other 32 are byte-identical**, including comment stripping on
well-formed lines (`model: a.xml # note about b.xml` was never affected — the comma is what
made the difference).

Six are the defect being fixed. Note that three of the five phantom-file lines are
consumers the issue did not name: it reported the two `model` spellings, but `mutant`,
`experiment: … data:` and the `data =` key have the identical defect, which is the
argument for fixing the token rather than each declaration.

| line | before | after |
|---|---|---|
| `model: a.xml, # note about b.xml` | `{'a.xml', '# note about b.xml'}` | parse error + `model` format hint |
| `model = a.xml : d.exp, # note about e.exp` | `{'d.exp', '# note about e.exp'}` | parse error + `model` format hint |
| `mutant = a.xml m1 x*2.0 : d.exp, # note about e.exp` | `{'d.exp', '# note about e.exp'}` | parse error + `mutant` format hint |
| `experiment: e, data: d.exp, # note about f.exp` | `{'d.exp', '# note about f.exp'}` | parse error + `experiment` format hint |
| `data = d.exp, # note about f.exp` | `['d.exp', '# note about f.exp']` | parse error |
| `model = a.xml : none # note about e.exp` | `{'none # note about e.exp'}` | `{}` — `none` finally means none |

Three are the narrowing, and they are the evidence that the two spellings now agree:

| line | before | after |
|---|---|---|
| `model: a#b.xml` | accepted | parse error |
| `model = a#b.xml : d#e.exp` | accepted | parse error |
| `experiment: …, measurement_params: a#b.tsv` | accepted | parse error |

## Consequences

* **A dangling comma before a comment is now a parse error rather than a phantom file.**
  `model: a.xml, # b.xml` is what a user writes when commenting out the second entry of a
  list and leaving the comma behind — a plausible editing state, and now a hard refusal.
  This is the right trade: the refusal carries the `model` format hint, which spells out the
  line's whole legal shape, and it arrives at parse time pointing at the line the user just
  edited, instead of arriving from the SBML loader pointing at a file that does not exist.
  The fix is to delete the comma, which is what they meant.
* **`: none` means none.** The `Or`-longest-match interaction is closed as a side effect
  rather than by special-casing `nonetoken`, because the cause was never the alternation.
* **The invariant is now stated where it is enforced.** Both the token comment and the
  ADR-0116 guard comment say which half each one closes, so the next person to widen a file
  token can see that the `#` boundary is load-bearing for two independent failures.
* **Not addressed here:** a malformed `data = …` line reports *"data is not a valid
  configuration key"*, which is wrong — `data` is valid, just narrow (ADR-0050, `objective =
  callable` only). It has no branch in `parse.py`'s per-key format-hint chain. That predates
  this change and is filed separately rather than folded in.
