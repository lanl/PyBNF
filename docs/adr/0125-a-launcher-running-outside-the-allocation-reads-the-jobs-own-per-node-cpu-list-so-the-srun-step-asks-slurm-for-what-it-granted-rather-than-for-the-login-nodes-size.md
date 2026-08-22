# A launcher running outside the allocation reads the job's own per-node CPU list, so the `srun` step asks SLURM for what it granted rather than for the login node's size (issue #642)

**Status: Accepted and implemented (2026-08-22).** The `srun` launcher (#614,
`cluster_type = slurm-srun`) sizes its worker pool by how many CPUs the job was granted on a node,
and asks SLURM for that many CPUs per task. It worked that number out from `$SLURM_CPUS_ON_NODE`,
which SLURM sets only inside a job step running on an allocated node. Launched from a shell that
is not on one -- which is where `salloc` leaves the user on many clusters -- it fell through to the
size of the machine it was running on, asked for that, and SLURM refused the step. This ADR makes
the count come from the allocation in that case too.

## The problem

The reporter's cluster documents `salloc` as the way to open an interactive allocation, and
`salloc` returns a shell **on the login node** while the allocation is held on a separate compute
node. That is the shell PyBNF is meant to run in: it holds the allocation, so `srun` starts a step
inside it rather than queuing a new job, which is what `docs/cluster.rst` tells the user to do and
what `require_slurm_allocation` checks for.

In that shell the launcher failed immediately:

    srun: error: Unable to create step for job NNNNN: More processors requested than permitted

`Cluster.cpus_per_node` prefers `$SLURM_CPUS_ON_NODE`. That variable is a **step** variable: SLURM
publishes it to a process running on a node the job holds, and not to a login-node shell, where it
is simply absent. The two remaining sources are both descriptions of the machine asking --
`dask.system.CPU_COUNT`, and the whole processor count -- and on the login node they describe the
login node, which is not in the allocation at all and is typically several times larger than what
the job holds. So a job granted 20 CPUs was sized as though it held 128, and `--cpus-per-task 128`
is a request SLURM refuses outright. No worker started, and the fit stopped.

`$SLURM_JOB_CPUS_PER_NODE` was already right in that shell, and PyBNF was already reading it:
`per_node_cpus` (#617) uses it to size each machine separately. It is a **job** variable, set
wherever the job's environment reaches, so it survives the trip to the login node. The default
equal-size path read it, confirmed every machine was the same size -- and then threw the counts
away and had `srun_worker_command` derive the number again from the environment.

This is not a variant of #616. That issue was about a count that described the *machine* instead of
the *job* while PyBNF was running inside the job. Here PyBNF is not inside the job's nodes at all,
so there is no local number that can be right, and the only usable answer is the one the scheduler
published about the allocation.

## The decision

### The allocation's own per-node list is a source of the single count

`cpus_per_node` gains a source between `$SLURM_CPUS_ON_NODE` and the two machine-level numbers: the
smallest entry of `$SLURM_JOB_CPUS_PER_NODE`. The order matters and is deliberate. Inside the
allocation, `$SLURM_CPUS_ON_NODE` stays preferred, exactly as #616 decided, so a launch from an
allocated node reads what it always read and nothing about it changes. The new source is reached
only when that variable is absent -- which is the case this issue is about -- and there it is
strictly better than the two below it, which describe a machine outside the allocation.

The **smallest** entry is the one taken, because this is the one-number answer: it sizes a pool
started on every machine, and it is what a single `srun` step asks for on every machine in it.
Asking for fewer CPUs than a machine holds costs speed; asking for more than the smallest machine
holds is refused outright, which is the failure being fixed. So when the entries differ the safe
direction is down. (Sizing each machine on its own is `per_node_cpus`, which the default `srun`
path already uses; this single number is the fallback and the SSH launcher's only option.)

Because both launchers size themselves through `cpus_per_node`, this also stops the SSH launcher
from starting a login node's worth of worker processes on each allocated machine when it is
launched from the login node.

### The layout hands its counts on rather than having them derived again

`srun_worker_layout` reads the per-node list before it decides which command to build. In the
equal-size case it now passes that count (and the phrase naming where it came from, for the log)
into `srun_worker_command`, which uses it verbatim; the command derives a count for itself only
when no caller supplied one. This is what the issue proposed, and it is worth doing on its own
terms even with the source list fixed: a number that has already been read from the allocation
should not be re-read from the environment, where a different variable can answer differently.
A `$SLURM_CPUS_ON_NODE` left over from an earlier allocation -- including one exported by hand as
the workaround for this issue -- no longer decides the size of a later run.

The argument list is otherwise untouched, so an allocation launched from inside itself builds the
same `srun` command it built before, which the existing regression test still pins.

### `parallel_count` keeps its single-count cap

Setting `parallel_count` still builds one step with an even split, and its CPU request is still
capped by the single `cpus_per_node` number so that a deliberately oversubscribed count starts all
the workers the user asked for without the step being refused. What changed is what that number
is: from the login node it now describes the allocation, so the cap does its job instead of being
measured against a machine that is not in the run. Making the cap per-machine is the limitation
ADR-0124 recorded as needing its own design; it is now tracked as #643 and stays there rather than
being folded in here.

## Consequences

* **The reported failure cannot happen from a count PyBNF chose.** A launch from the login-node
  shell `salloc` opens asks SLURM for what the job holds. The workaround the reporter found --
  exporting `SLURM_CPUS_ON_NODE` by hand -- is no longer needed, and no longer has an effect on a
  later allocation if it is left set.
* **A launch from inside the allocation is unchanged.** `$SLURM_CPUS_ON_NODE` is still preferred
  where SLURM sets it, so the command built there, and the number in the log, are what they were.
* **The SSH launcher is fixed too, for free.** It reads the same single count, so a `-t slurm` run
  started from a login node no longer sizes each remote machine by the login node's processors.
* **The log still names the source.** The new phrase names `$SLURM_JOB_CPUS_PER_NODE`, so a user
  who sees an unexpected count can still trace which number PyBNF believed and where it came from.
* **What was verified where.** The constructed `srun` argument list is the oracle, as it was for
  #614 and #617: a login-node environment (no `$SLURM_CPUS_ON_NODE`, a job list of 20, machine-level
  numbers of 128) is stood up in the test and the command must ask for 20, on the default path and
  on the `parallel_count` path. That SLURM then accepts the step is what the reporter's cluster
  verifies.

## Alternatives considered

* **Only route the equal-size path through the per-node counts** (the issue's suggested fix, taken
  on its own). Rejected as half of the fix: it leaves `parallel_count` and the SSH launcher reading
  the login node, and leaves the fallback inside `per_node_cpus` -- reached when SLURM's list cannot
  be lined up with the machines -- reading it too. It is kept as the *other* half, because handing
  on a count already read is better than re-deriving it however good the source list is.
* **Prefer `$SLURM_JOB_CPUS_PER_NODE` over `$SLURM_CPUS_ON_NODE` everywhere.** Rejected: on a
  mixed allocation launched from an allocated node, the step variable describes *this* machine and
  the job list has to be reduced to one number to compete with it. Reordering would change what
  every existing SLURM run reads in order to fix a case where the step variable is not set at all.
* **Refuse to run when no allocation-derived count is available.** Rejected: PyBNF has a usable
  answer in every case that reaches the machine-level sources -- a single-machine allocation of a
  whole node, a run outside SLURM entirely -- and refusing would break runs that work today.
* **Detect the login node and warn.** Rejected as the wrong shape: there is no reliable test for
  "this machine is not in the allocation" that is better than simply reading the number SLURM
  published, and once the right number is read there is nothing to warn about.
