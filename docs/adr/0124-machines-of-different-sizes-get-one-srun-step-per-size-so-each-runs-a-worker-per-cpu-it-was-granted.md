# Machines of different sizes get one srun step per size, so each machine runs one worker per CPU it was granted rather than a count borrowed from another machine (issue #617)

**Status: Accepted and implemented (2026-08-21).** When PyBNF started workers across several
machines it sent the same worker count to every one. On a cluster whose machines differ in size,
that single count is wrong for some of them. This ADR changes the `srun` launcher (#614,
`cluster_type = slurm-srun`) to size each machine by what SLURM granted it, starting one `srun`
job step per distinct machine size. The SSH launcher is unchanged, and so is every allocation
whose machines are all the same size.

## The problem

`Cluster.cpus_per_node` returns one number, and both launchers used to start that many workers on
every machine. The number describes the machine PyBNF is running on. On a cluster whose machines
are all the same size that is right for all of them, which is the case the launcher was first built
and tested against.

The reporter's cluster is not that case. It has a single queue whose machines differ in processor
count by more than a factor of two, and separate requests land on unequal machines. One count for
all of them is then too many workers for a small machine, where the extra workers compete for its
processors, and too few for a large machine, which runs fewer workers than it has processors and
sits partly idle. Neither failure stops the run or prints anything wrong. The fit is simply slower
than the allocation could have made it, in a way nothing in the log accounts for.

This could not be fixed until a launcher existed that starts each machine's workers separately.
`dask ssh` takes one worker count for all hosts and has no way to say a different number per host,
which is why this fix cannot be built on the SSH launcher. The `srun` launcher, added in #614, runs
`srun` itself and can run it more than once. ADR-0122 recorded this as the reason that launcher
would be where #617 was fixed: "`srun` ... can be invoked per node group, so this launcher is what
that fix will be built on."

## The decision

### One `srun` job step per distinct machine size

A single `srun` step cannot express different worker counts on different machines. `--cpus-per-task`
is one value for the whole step, and it is not decoration: under `task/cgroup` binding a task that
asks for fewer CPUs than the workers it forks is confined to the CPUs it asked for, which quietly
serializes the node (ADR-0122 documents this as the reason the count is requested at all). So a step
that asked for the small machine's CPU count would throttle the large machines in the same step, and
a step that asked for the large machine's count would be refused on the small ones, which do not have
that many.

The machines are therefore grouped by how many CPUs each was granted, and each group gets its own
`srun` step with its own `--cpus-per-task` and `--nworkers`, naming its machines with `--nodelist`.
A run on two 40-processor machines and one 96-processor machine becomes two steps: 40 workers on each
of the first two, 96 on the third. The steps run on disjoint machines, which SLURM allows to run at
the same time, so this does not serialize the bring-up.

Grouping by size, rather than one step per machine, is what keeps the common case unchanged. An
allocation whose machines are all one size is a single group, so it runs as exactly one `srun` step,
the same command the launcher built before this change. Only a genuinely mixed allocation starts
more than one step, and then only as many as there are distinct sizes.

### Where the per-machine counts come from

`Cluster.per_node_cpus` reads `$SLURM_JOB_CPUS_PER_NODE`, which SLURM publishes as a compressed
per-node list (for example `40(x2),96` for two 40-processor machines and one 96-processor machine)
in the same order as the node list `scontrol show hostname` returns. Expanding it gives one count
per machine, lined up with the names PyBNF already read.

This is the granted allocation, not the size of the machine, for the same reason the single-count
path reads `$SLURM_CPUS_ON_NODE`: the count is not only how many workers to start, it is how many
CPUs the step asks SLURM for, and a number taken from the whole machine would be refused. The
existing per-worker cap (`--cpus-per-task` no larger than what the job holds) is unchanged.

### The fallback is the old behavior, said out loud

If `$SLURM_JOB_CPUS_PER_NODE` is unset, does not parse, or does not have exactly one entry per
machine in the allocation, PyBNF cannot line a count up with each machine. Rather than guess, it
falls back to the single `cpus_per_node` count for every machine, which is what the launcher did
before this change, and logs a warning naming which of those three reasons applied. A user on a
mixed cluster is expecting each machine to be sized on its own, so the one case where that did not
happen is worth a line in the log rather than silent.

### Readiness now waits for every worker, not just the first

The `srun` launcher used to treat one registered worker as the readiness signal. With more than one
step that is not enough: a second step whose placement failed would go unnoticed as long as the
first step's workers registered. The wait now counts all of the workers the steps should produce
and returns when that many have registered, and it watches every step's process while it waits, so
a step that dies is reported with that step's own log rather than masked by another that succeeded.

This is a deliberate strengthening of the readiness bar on this path, not only for the multi-step
case. It also closes on the `srun` launcher the same gap #200 describes for the SSH launcher, where
a bring-up that produced fewer workers than asked for was accepted as long as one arrived. The SSH
launcher's own readiness check is unchanged.

### `parallel_count` is left as an even split

**Superseded by issue #643 / ADR-0126:** the override is now split in proportion to machine size on
a mixed allocation, one `srun` step per size, so it no longer asks a smaller machine for the larger
machines' share. The rest of this section records why #617 left it alone at the time.

Setting `parallel_count` overrides the automatic sizing with a total number of workers split evenly
across the machines, exactly as the SSH launcher has always done. This change does not make that
override per-machine. A user who names a total is asking for that many workers, and dividing an
explicit total by machine size is a different decision from sizing an automatic run, with its own
question of what the total means on machines that cannot hold an equal share. So the override keeps
its single-step, even-split behavior, and only the default (auto-sized) path became per-machine.

## Consequences

* **A mixed-size allocation is used as fully as the launcher can use it.** Each machine runs a
  worker per CPU it was granted, so small machines are not oversubscribed and large machines are
  not left idle. The per-machine arrangement is in the log, so an unexpected count can be traced.
* **The common case is byte-for-byte unchanged.** An allocation of same-size machines, and any run
  with `parallel_count` set, builds exactly the single `srun` command it did before, which a
  regression test pins against the pre-change argv.
* **A mixed allocation writes more than one worker log.** Each step writes its own
  (`dask_workers.log`, `dask_workers_2.log`, and so on), because concurrent steps writing one file
  would interleave and truncate each other. The readiness error names the logs it read.
* **What was verified where.** The commands PyBNF builds for the grouped case, the grouping itself,
  the fallback, and the stronger readiness count were exercised on one machine with stand-ins for
  `srun` and `scontrol`, in the same way #614 was: the constructed command string is the oracle.
  That SLURM places concurrent per-group steps as intended on a real heterogeneous allocation is
  what the reporter's cluster verifies, exactly as it did for #614.
* **A known limitation on the `parallel_count` path is left in place.** Because that override splits
  a total evenly and requests `--cpus-per-task` for the per-node share, a share larger than a
  machine smaller than the one PyBNF runs on can be refused by SLURM on that machine. This predates
  #617 and is out of its scope. It is recorded here rather than fixed; if it needs fixing it needs
  its own design, and a tracking issue, rather than being folded into this change. That issue is
  #643, now fixed: the override is split in proportion to machine size, one step per size (ADR-0126).

## Alternatives considered

* **One `srun` step per machine.** Rejected: it is more steps than the problem needs, and it makes
  even a same-size allocation, the common case, stop building the single command that is already
  tested and known good. Grouping by size gives each distinct size its own binding while leaving the
  homogeneous case as one step.
* **Make `parallel_count` per-machine too.** Rejected as out of scope. Sizing an automatic run by
  what each machine holds is a clear rule; splitting a user's explicit total by machine size raises
  a separate question of what the total means when machines cannot take equal shares, and the SSH
  launcher's even split is the behavior users of `parallel_count` have today.
* **Keep the single-worker readiness check.** Rejected: with more than one step, a step whose
  placement failed would be masked by another that succeeded, turning a half-empty cluster into a
  fit that runs slowly for no visible reason. Waiting for the full count is what makes the multi-step
  bring-up safe.
* **Fail rather than fall back when `$SLURM_JOB_CPUS_PER_NODE` cannot be lined up with the machines.**
  Rejected: the launcher has a correct, if less precise, thing to do (size every machine the same, as
  it did before), so refusing to run would be worse than doing that and saying so. The warning makes
  the degraded case visible without stopping the fit.
