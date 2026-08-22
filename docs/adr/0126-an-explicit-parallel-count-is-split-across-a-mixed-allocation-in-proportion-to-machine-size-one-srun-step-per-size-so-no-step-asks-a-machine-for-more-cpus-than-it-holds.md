# An explicit `parallel_count` is split across a mixed allocation in proportion to machine size, one `srun` step per size, so no step asks a machine for more CPUs than it holds (issue #643)

**Status: Accepted and implemented (2026-08-22).** Setting `parallel_count` gives the `srun`
launcher (#614, `cluster_type = slurm-srun`) a total number of workers to start over all machines.
The launcher split that total evenly and asked SLURM, in a single job step, for the per-machine
share on every machine. On an allocation whose machines are not all the same size, the even share
can be more than a smaller machine was granted, and SLURM refuses the whole step. This ADR splits
the total in proportion to each machine's size and starts one step per size, the way the automatic
sizing already works, so each step asks only for what its machines hold. It supersedes the decision
ADR-0124 recorded to leave this path an even split.

## The problem

`parallel_count` is a total, and the `srun` launcher divided it evenly:
`n_per_node = ceil(parallel_count / node_count)`. One `srun` step then asked SLURM for that many
CPUs per task with `--cpus-per-task`. That request has to be satisfiable on *every* machine the
step runs on, but it was capped against a single CPU count rather than against the smallest machine
in the allocation. On a mixed allocation the even share can exceed what the smaller machines were
granted, and SLURM refuses the step:

    srun: error: Unable to create step for job NNNNN: More processors requested than permitted

On a 96-CPU machine and a 40-CPU machine with `parallel_count = 136`, the even share is 68 per
machine. Launched from the 96-CPU machine the cap is 96, so the step asks for 68 CPUs on both
machines, and the 40-CPU machine cannot supply them. No worker starts, and the fit stops.

The automatic path does not have this problem. #617 (ADR-0124) gave it one `srun` step per distinct
machine size, each asking only for what those machines hold. Only the explicit-override path was
left as an even split. #642 (ADR-0125) did not change it either: it changed which single number the
even split's cap is measured against, so that the number describes the allocation rather than the
machine PyBNF was launched from, which is a different bug.

ADR-0124 recorded this deliberately, under "`parallel_count` is left as an even split", and asked
for it to be tracked separately rather than folded into #617, because a fix changes what an
existing config key means on a mixed allocation and wanted its own decision. This is that decision.

## The decision

### On a mixed allocation, split the total in proportion to machine size

The automatic path already groups the machines by how many CPUs each was granted and gives each
group its own `srun` step, so a smaller machine and a larger one are never in the same step and
never bound to the same `--cpus-per-task`. `parallel_count` now uses the same grouping. For a size
group whose machines were each granted `cpus` CPUs, with the allocation holding `total_cpus` across
all machines:

    workers per machine in the group = max(1, ceil(cpus * parallel_count / total_cpus))
    CPUs the step asks for            = max(1, min(workers per machine, cpus))

A group holds `cpus / total_cpus` of the allocation's processors, so it gets that share of the
requested total. Each step's CPU request is capped at its own group's machine size, so SLURM
accepts every step; an oversubscribed `parallel_count` still starts every worker asked for while
the request stays satisfiable, the same contract the single-step path already had.

On the 96-CPU and 40-CPU example with `parallel_count = 136`, which is exactly what the allocation
holds, this is one worker per CPU: 96 workers asking for 96 CPUs on the larger machine, 40 asking
for 40 on the smaller. Both steps are accepted, and both machines are fully and correctly bound.

### `ceil`, and a floor of one, so the homogeneous case is the even split exactly

The rounding is the same `ceil` the even split has always used, and this is not a coincidence: when
every machine is the same size, `cpus / total_cpus` is `1 / node_count`, so the per-machine count is
`ceil(parallel_count / node_count)`, the number the even split produces. The per-size split is the
generalization of the even split, not a different rule bolted on beside it, and a homogeneous
allocation is left running the single even-split step it ran before, byte for byte. Because `ceil`
rounds each machine's share up, the realized total can sit just above the request, exactly as the
even split's total already can (`ceil(5 / 3) * 3 = 6`). The floor of one guarantees every machine in
the allocation runs at least one worker even when its proportional share rounds to nothing.

### Only the mixed case changes; the layout reads the sizes once and branches

`srun_worker_layout` now reads the allocation's own per-node CPU list first, for both the automatic
and the explicit paths, and asks a single question: are the machines all one size? If they are, it
builds the one step it always built, with the count coming from `parallel_count` or from the grant
as before. If they are not, it builds one step per size, with each machine's worker count coming
from the grant (automatic) or from the proportional split (explicit). Reading the list first also
means the explicit path hands the count it read straight to the step rather than the step reading
the environment a second time, which is the property #642 relied on: a `$SLURM_CPUS_ON_NODE` that
does not describe this job no longer sizes the run.

## Consequences

* **The reported failure cannot happen from a count PyBNF chose.** Every step asks only for what
  its own machines hold, so `parallel_count` on a mixed allocation no longer produces a request
  SLURM refuses. That SLURM then places the concurrent per-size steps as intended on a real
  heterogeneous allocation is what the reporter's cluster verifies, as it did for #614 and #617.
* **What `parallel_count` means on a mixed allocation changed.** It was an even split; it is now a
  size-weighted split. This is the change ADR-0124 declined to make in passing. On a homogeneous
  allocation, which is the common case, nothing changed. The realized per-machine arrangement and
  the total are written to the log, one line per size, so an unexpected count can be traced.
* **A mixed allocation with `parallel_count` writes more than one worker log**, the same way the
  automatic mixed path does (`dask_workers.log`, `dask_workers_2.log`, and so on), because
  concurrent steps writing one file would interleave and truncate each other.
* **The common case is byte-for-byte unchanged.** A homogeneous allocation, with or without
  `parallel_count`, builds exactly the single `srun` command it built before, which a regression
  test pins. The automatic mixed path (#617) is unchanged: the per-group command builder defaults
  its worker count to one per granted CPU, so a caller that does not pass a count gets the old argv.
* **What was verified where.** The constructed `srun` argument list is the oracle, as it was for
  #614, #617, and #642. Tests stand up a mixed allocation (`40(x2),96`, and the reporter's `96,40`)
  with `parallel_count` set and require each step to ask only for what its machines hold, on the
  ordinary and the oversubscribed cases, and require the homogeneous case to stay one even-split
  step.

## Alternatives considered

* **Cap the single even-split step at the smallest machine.** The step would be accepted, but a
  task on a larger machine would fork the even share while `--cpus-per-task` asked only for the
  smallest machine's count, and under `task/cgroup` binding that task is confined to the CPUs it
  asked for (ADR-0122 documents this confinement as the reason the count is requested at all). The
  larger machines would run partly serialized and the smaller machine oversubscribed, so the fit
  would run slowly for no visible reason. Rejected: it trades a loud failure for a quiet one.
* **Distribute the total by a largest-remainder method so it sums to `parallel_count` exactly.**
  Rejected as more machinery than the case needs. It would also have to give two same-size machines
  different counts to spend the last unit, which breaks "one step per size" and would need a step
  per machine. `ceil` keeps same-size machines identical, which is what lets them share one step,
  and the even split already does not hit the total exactly, so matching it precisely buys nothing.
* **Leave the override an even split and only document the limitation.** This is what ADR-0124 did,
  on the ground that the fix wanted its own decision. That decision is this ADR; leaving it undone
  keeps a config key that fails outright on the mixed allocations #617 exists to support.
