# Lesson 23 — Checkpoint & resume (`-r`)

**Feature:** the `--resume` CLI flag + automatic backups · **Difficulty:** ★☆☆

Long fits get interrupted: a cluster job hits its wall-time, a laptop sleeps, a run
just runs over budget. PyBNF checkpoints every run so you can continue instead of
starting from scratch — and so you can push a finished run further without redoing
the work you already paid for.

## How it works

As a fit runs, PyBNF pickles the **entire algorithm state** — the population, the
trajectory, the best-so-far, even the RNG — to `alg_backup.bp` in the `output_dir`,
refreshed every `backup_every` iterations. When the run finishes, that file is
renamed to `alg_finished.bp`. Either one is a complete restart point.

```
output/resume_fit/
├── alg_backup.bp      # present while a run is in progress
└── alg_finished.bp    # the same file, renamed once the run completes
```

## Resuming

```bash
# An interrupted run left alg_backup.bp behind — finish it:
pybnf -c resume_fit.conf -r

# A finished run you want to push further — add 10 more iterations:
pybnf -c resume_fit.conf -r 10
```

`-r` reloads the checkpoint and **continues the same trajectory**: the iteration
counter, the population, and the best-so-far all carry over. So `-r 10` on this
20-iteration fit produces the same 30-iteration result you'd get from one
uninterrupted run — the objective at the end is no worse than where it stopped, and
usually much better (here it drops the objective by more than an order of magnitude).

Two rules worth knowing:

- On an **interrupted** run (`alg_backup.bp` present), a bare `-r` just finishes it.
- On a **finished** run (`alg_finished.bp` present), a bare `-r` is an error — the
  run is already done, so you must say how many *more* iterations to add: `-r N`.

If neither checkpoint file is in the `output_dir`, `-r` has nothing to resume and
reports so. (A fixed `random_seed` in the conf makes the whole run reproducible and
lets the checkpoint restore the RNG exactly, so a resumed run is deterministic
rather than re-seeded.)

## Try it

```bash
pybnf -c resume_fit.conf -o        # 20 iterations -> alg_finished.bp
pybnf -c resume_fit.conf -r 10     # resumes, runs 10 more, ends lower
```

The second command prints `Resuming a fitting run` and continues numbering from
where the first stopped.

## The test

[`tests/test_tutorial_resume.py`](../../../tests/test_tutorial_resume.py) (slow tier,
because it drives the real CLI twice) runs the fit to completion, then resumes it
with `-r 10`, and asserts the resumed run announces itself, ends at no worse an
objective than it stopped at, and recovers the true `(k, A0)`.
