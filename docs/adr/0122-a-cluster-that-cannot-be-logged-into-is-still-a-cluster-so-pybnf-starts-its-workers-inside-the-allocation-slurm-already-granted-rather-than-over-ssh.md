# A cluster that cannot be logged into is still a cluster, so PyBNF starts its workers inside the allocation SLURM already granted rather than over SSH (issue #614)

**Status: Accepted and implemented (2026-08-21).** PyBNF had exactly one way to run a fit across
several machines: run `dask-ssh`, which logs in to every node. On a cluster whose nodes authenticate
to each other by host-based or Kerberos SSH, that login cannot succeed -- not with more
configuration, not with SSH keys -- so multi-machine fitting was unavailable there outright. This
ADR adds a second launcher, selected by `cluster_type = slurm-srun`, that starts the workers with
SLURM's own `srun` and never authenticates anything. The SSH launcher is unchanged and remains the
default for `-t slurm`.

## The problem

`Cluster.setup_cluster` starts workers by running `dask-ssh`, and `dask-ssh` opens its connections
with **paramiko**, not with the operating system's `ssh`. paramiko offers two ways to log in: a
public key, or a password. Clusters commonly use neither:

* **Host-based authentication**, where the machines are configured to trust each other and no user
  credential is involved at all. paramiko has no support for it.
* **Kerberos / GSSAPI**. paramiko can do this; dask never turns it on.

The user-visible shape of the failure is what makes it costly: on the cluster in #614, logging in by
hand from the same shell works.

```console
$ salloc -N 2
$ ssh OTHER hostname
OTHER                                    # succeeds

$ python -c "import paramiko; c = paramiko.SSHClient(); \
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy()); c.connect('OTHER')"
paramiko.ssh_exception.AuthenticationException: Authentication failed.
```

The second command is the login PyBNF attempts. So the run stops before a single simulation starts,
while every check the user can think to make says SSH is fine. `docs/cluster.rst` answered this
symptom with instructions for creating SSH keys, which cannot help: the cluster is not asking for a
key, and on a host-based cluster there is no per-user credential to install.

Nothing in that chain is PyBNF's to fix. paramiko does not implement host-based authentication;
enabling its GSSAPI support is dask's call, not ours, and would still leave host-based clusters out.
The fix has to be a way of starting workers that does not log in at all.

## The decision

### A second launcher, chosen by a new `cluster_type` value

`cluster_type = slurm-srun` (also accepted: `slurm_srun`, `slurmsrun`, `srun`) selects a launcher
that does the same three things a cluster bring-up always does, without an SSH login anywhere:

1. start a dask **scheduler** on the node PyBNF is running on, told to write a connection file;
2. start the **workers** with `srun`, one task per node of the allocation, each reading that file;
3. **connect** through the same file -- which PyBNF already supported, via `scheduler_file`.

No credential is involved in step 2 because SLURM granted the allocation before PyBNF started. That
is the entire argument for the design: the authorization the SSH login was trying to re-establish
already exists, and `srun` is the interface to it.

This is a new value rather than a change to `slurm` so that **anyone whose SSH setup already works
is unaffected**. Both spellings reach the same node-list code (`read_node_names`); only the way the
workers are started differs. One ordering hazard is load-bearing and is pinned by a test:
`re.match('slurm', 'slurm-srun')` succeeds, so the srun test has to come first, or the new value
would silently take the SSH path it exists to avoid.

### The scheduler runs here; only the workers go through SLURM

The scheduler is an ordinary subprocess of the PyBNF process, on the node PyBNF is already running
on. Nothing is gained by placing it with `srun` -- it is not a compute task, it must outlive
individual worker failures, and the process PyBNF wants to be able to terminate at teardown is
better held directly than through a job step. `srun` is used for exactly the thing it is needed for:
placing processes on machines this process cannot log in to.

Both commands are run as `sys.executable -m dask ...` rather than as a bare `dask` on `PATH`. A
launcher whose purpose is to remove ambiguity about *how* a worker starts should not reintroduce
ambiguity about *which* environment starts it, and PyBNF already requires the shared filesystem that
makes this interpreter path valid on every node. (It also sidesteps the console-script rename behind
issue #615 outright: `dask-scheduler` and `dask-worker` no longer exist as separate scripts, and
`python -m dask` does not depend on which of them a given dask version installs.)

### Readiness is polled on a real signal, never slept through

The SSH launcher waits a fixed ten seconds and hopes. The srun launcher waits on the two facts it
actually needs, and watches the process it started while it waits:

| Wait | Signal | Failure it catches at once |
| --- | --- | --- |
| scheduler up | the connection file parses as JSON with an `address` | scheduler exited -- port in use, bad interpreter |
| workers up | at least one worker registered with the scheduler | `srun` exited -- bad flags, request larger than the allocation |

The file is written in place rather than renamed into place, so a reader can catch it half-written;
requiring it to *parse* is what makes its appearance a readiness signal rather than a race. A
scheduler file left over from an earlier run is deleted before the scheduler starts, so its
reappearance proves this scheduler started rather than being indistinguishable from history.

The worker check is the one that cannot be dropped. Connecting to our own scheduler always succeeds
-- a scheduler with no workers is a perfectly good scheduler -- so a failed placement would
otherwise surface as a fit that submits jobs and never gets one back, with the reason sitting
unread in `srun`'s output. Both failures quote that output in the error.

### Preconditions are refused, not discovered

Outside an allocation, `srun` does not place a task: it *submits a job* and waits to be granted one.
That failure would look like PyBNF hanging with no output, possibly for hours, so a missing
`$SLURM_JOB_ID` is refused up front with a message naming the remedy (start PyBNF from the shell
that holds the allocation -- the one `salloc` opened, or the `sbatch` script -- since a separate
login into an allocated node does not inherit it).

### The CPU request tracks the worker count

The workers are placed as `--ntasks-per-node 1 --cpus-per-task N`, one `dask worker` launcher per
node forking `N` single-threaded workers. `--cpus-per-task` is not decoration: under `task/cgroup`
binding, a task that took the default single CPU confines every process it forks to that one CPU,
which would quietly serialize a whole node while looking like a healthy cluster. So the request
tracks the number of workers, capped at what the job actually holds -- SLURM refuses a request
larger than the allocation, and a deliberately oversubscribed `parallel_count` (which the SSH
launcher has always allowed) should not become a hard failure.

Where that count comes from differs between the launchers, and deliberately so. This path reads
`$SLURM_CPUS_ON_NODE` -- what the job was *granted* -- because it does not merely count workers with
that number, it asks SLURM for that many CPUs, and a number taken from the whole machine would be
refused. The SSH launcher still uses `multiprocessing.cpu_count()`; that it does so is issue #616,
and it has to be fixed there rather than here.

### `scheduler_file` names an output under this launcher

Everywhere else, `scheduler_file` means *attach to a cluster someone else brought up* -- PyBNF starts
nothing. Under the srun launcher PyBNF starts the scheduler that writes the file, so the key
instead chooses **where** it is written, defaulting to `dask_scheduler.json` inside `output_dir`.
The two readings are not in tension: with the launcher selected the user has explicitly asked PyBNF
to bring the cluster up, so the file can only be an output. What PyBNF wrote it also removes at
teardown, since a connection file naming a scheduler that is shutting down is exactly what the next
run would mistake for a live cluster. A file PyBNF did not write is never touched.

## Consequences

* **Multi-machine fitting is possible on host-based and Kerberos clusters**, which is the whole
  point. The reporter confirmed the approach on the cluster in #614: two workers started on two
  separate machines on every attempt.
* **What was verified where.** The behavior above -- both bring-up steps, both readiness checks,
  the failure paths, and teardown -- was exercised end to end on one machine with stand-ins for
  `srun` and `scontrol` on `PATH`: a real dask scheduler, two real workers, a real task computed
  through the client, both processes gone after teardown. That validates the commands PyBNF
  constructs and its own logic; it does not validate SLURM's placement semantics, which is what the
  reporter's cluster test covers.
* **Nothing changes for a run that does not ask for it.** `-t slurm`, `scheduler_node` /
  `worker_nodes`, an attached `scheduler_file`, and every local run construct exactly the calls they
  did before. The srun branch is entered only by the new `cluster_type` values.
* **Two logs, in the output directory.** `dask_scheduler.log` and `dask_workers.log` are the only
  record of what those processes said, since neither has a terminal. They are files rather than
  pipes because both processes outlive the call that starts them and an undrained pipe would
  eventually deadlock the writer.
* **A failed bring-up leaves nothing running.** A constructor that raises never becomes a `Cluster`,
  so nobody else can tear it down; the srun path stops what it started on the way out.
* **Per-machine worker counts become expressible.** `dask ssh` takes one worker count for all hosts,
  which is why #617 cannot be fixed on that path; `srun` is invoked per allocation and can be
  invoked per node group, so this launcher is what that fix will be built on.
* **`#398`'s complaint is not fixed, only not repeated.** The SSH launcher's two fixed ten-second
  sleeps are untouched. The new path has none.

## Alternatives considered

* **Turn on paramiko's GSSAPI support in dask.** Rejected: it is an upstream change we do not
  control, it would not help host-based clusters at all (paramiko cannot do host-based
  authentication), and it would leave PyBNF's multi-machine support depending on a login that the
  cluster is configured not to need.
* **Shell out to the system `ssh` instead of paramiko.** This would honor whatever the cluster's own
  SSH is configured to do, including both failing modes. Rejected because it means reimplementing
  `dask ssh` -- remote process launch, log forwarding, teardown, failure detection -- inside PyBNF,
  to solve a problem SLURM solves in one command for the clusters that have this problem in the
  first place.
* **Use `dask-jobqueue`'s `SLURMCluster`.** It submits *new* jobs and scales them, which is a
  different execution model from PyBNF's (run inside the allocation the user already holds), would
  change what a `-N 4` allocation means for a fit, and adds a dependency to fix a login bug.
* **Make srun the only SLURM launcher.** Rejected: users whose SSH setup works today would be
  migrated onto a path with different placement semantics and a hard requirement to be the only job
  step in the allocation, for no benefit to them. #614 asks for a new value precisely so that they
  are unaffected.
* **Pass `--overlap` to `srun` so a second job step can always be created.** Rejected: the flag does
  not exist before SLURM 20.11, where passing it turns a working bring-up into an immediate failure.
  PyBNF launches exactly one job step, so the case it guards against is a user running another one
  concurrently -- documented, and named in the timeout message, rather than paid for on every
  cluster.
* **Let the workers keep the default single CPU per task and rely on `--cpu-bind=none`.** Rejected:
  where the binding comes from a cgroup, `--cpu-bind=none` does not widen it. Requesting the CPUs
  the workers need is the only request that is honored under both task plugins.
