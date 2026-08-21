# Every locally spawned dask worker is single-threaded, so a fit's thread policy no longer depends on whether `parallel_count` is set (issue #526)

**Status: Accepted and implemented (2026-07-30).** `pybnf/cluster.py` built its local dask client
two different ways: with `parallel_count` set it constructed `LocalCluster(n_workers=...,
threads_per_worker=1)`, and without it took dask's bare `Client()`, whose workers are
multi-threaded on any machine with more than four cores. Whether a worker process ran one thread or
several therefore depended on an unrelated key. This ADR collapses the two local branches into one
that always pins `threads_per_worker=1`; `parallel_count` chooses how many worker *processes* there
are and nothing else.

## The problem

The simulation backends hold process-wide state that is not advertised as thread-safe: a C++ engine
plus code generation with module-level caches. PyBNF has acted on that everywhere except the local
default:

* both `dask-ssh` branches pass `--nthreads 1` (the default one pairs it with
  `--nworkers <cores>`);
* `docs/cluster.rst` tells a user configuring dask by hand to "keep `nthreads` equal to 1 ...
  because the SBML simulator is not thread safe";
* ADR-0065 routes a *scored* Newton/KINSOL dose-response scan through the sequential per-point loop
  rather than the thread pool, because the sensitivity solve is not confirmed thread-safe;
* the explicit `parallel_count` branch pinned 1 deliberately.

Only the branch a user gets by configuring nothing let dask pick the thread count.

That difference is observable, and issue #525 is where it surfaced. bngsim 0.11.35's `sympy_to_c`
emits through a process-wide cached printer whose resolver it assigns and then clears, which is not
thread-safe; concurrent emissions inside one worker race on that attribute and the loser reports an
ordinary quotient as non-differentiable. Measured directly on the same arithmetic derivative:
**0 spurious refusals in 2000 serial calls, 14 in 4000 across 8 threads.** On the reporter's job
(`examples/real-world/Salazar-Cavazos-2019/egfr_simpull` switched to `job_type = trf`), no
`parallel_count` aborted the fit before the first start completed, while `parallel_count = 4` --
nothing else changed -- completed five TRF iterations with zero ragged dose points. The one
difference between those two runs was threads per worker.

So the default was the less safe of the two configurations, reachable only by *not* configuring
anything. ADR-0088 fixed the way that failure crashed (a ragged sensitivity set now stacks by
selector name instead of by position) and explicitly left this decision open.

## The decision

### One local branch, one thread policy

The `parallel_count is not None` test no longer selects between two client constructions. There is
a single local branch that builds `LocalCluster(**Cluster.local_cluster_kwargs(parallel_count))`,
and the policy lives in one static method:

```python
@staticmethod
def local_cluster_kwargs(parallel_count):
    kwargs = {'threads_per_worker': 1}
    if parallel_count is not None:
        kwargs['n_workers'] = parallel_count
    return kwargs
```

The two branches can no longer disagree about thread safety, because there is only one place that
decides. `parallel_count` is a process count; it never raises `threads_per_worker` above 1.

### `n_workers` is left to dask when `parallel_count` is unset

Given `threads_per_worker=1` and no `n_workers`, dask sets `n_workers = CPU_COUNT`. Delegating is
deliberate rather than computing `multiprocessing.cpu_count()` here: `dask.system.CPU_COUNT` honors
CPU affinity and cgroup quotas, so a run confined to 4 cores of a 64-core host gets 4 workers, not
64. (`setup_cluster` still uses `multiprocessing.cpu_count()` for `dask-ssh`, where the number
being computed is a remote node's core count anyway.)

**Superseded by issue #616:** the parenthesis above was the defect. A remote node's core count is
not what a *job* holds on that node -- a job granted 4 CPUs of a 128-processor node was told 128 --
so `setup_cluster` now takes its default from `Cluster.cpus_per_node`, which prefers what the
scheduler granted (`$SLURM_CPUS_ON_NODE`) and falls back to `dask.system.CPU_COUNT` before ever
reaching `multiprocessing.cpu_count()`. Both launchers now decide this in that one place.

**Total concurrency is unchanged.** Measured on a 6-core machine, before and after:

```text
OLD  Client()                                 workers 3 x 2 threads  = 6 concurrent jobs
NEW  LocalCluster(threads_per_worker=1)       workers 6 x 1 thread   = 6 concurrent jobs
     LocalCluster(n_workers=4, threads=1)     workers 4 x 1 thread   = 4 concurrent jobs
```

Six jobs still run at once by default. What changes is that each of them now has a process to
itself. The same holds generally: dask's default split (`nprocesses_nthreads`) always factors the
core count into `processes x threads`, so one thread per core is what both the old and the new
default deliver.

### No key to opt back into multi-threaded workers

Multi-threaded local workers were never a supported configuration -- they were what dask picked
when PyBNF said nothing -- so nothing is being taken away that a user could have relied on. Adding a
key would mean shipping a documented way to run the backends in a mode this project has said three
times is unsafe.

## Consequences

* **A default local fit is no longer exposed to intra-process races.** The failure mode of #525
  cannot occur at all in the default configuration, rather than being avoided by setting an
  unrelated key. Measured as an A/B on the reporter's own job -- `egfr_simpull` under
  `job_type = trf`, 8 concurrent starts, **no `parallel_count`**, three runs per side on a 6-core
  machine, the only difference being the local client PyBNF builds:

  | local client | exit code | ragged-column warnings |
  | --- | --- | --- |
  | this ADR (6 workers x 1 thread) | 0, 0, 0 | 0, 0, 0 |
  | previous default (3 workers x 2 threads) | 1, 1, 1 | 4, 4, 2 |

  Every previous-default run died in about five seconds with "Gradient-based fitting is not
  available for this fit; use a metaheuristic `job_type` instead", having lost the sensitivity
  column of a *scored* observable -- and a different one each time (`phosR_per`,
  `pY1173_percent`, `phosR_per`), which is the signature of a race rather than a property of the
  model. Every run under this ADR completed all 8 starts with no dropped column at all.
* **More processes, hence more memory.** Six single-threaded workers hold six copies of the models
  where three dual-threaded workers held three. The base simulation models are unpickled per
  evaluation and amortized at module scope (#415), so that cache is now per-core rather than per
  factor-of-the-core-count. Dask's `memory_limit="auto"` also divides host memory by the worker
  count, so each worker's share is smaller. `parallel_count` is the lever for both, and
  `docs/config_keys.rst` now says so.
* **Explicit `parallel_count` runs are byte-identical.** They already built exactly these kwargs;
  the constructed `LocalCluster(...)` and `Client(...)` calls are unchanged.
* **The log line now says what was built.** "Creating a local client with default parallel count"
  reported nothing observable; it is now either "one single-threaded worker per available core" or
  "manually set to N single-threaded workers".
* **The invariant is tested, not just implemented.** `tests/test_cluster.py` asserts
  `threads_per_worker == 1` across every local configuration (`parallel_count` unset, 1, 4, 36),
  and asserts that the local default and the `dask-ssh` default agree on one thread per worker --
  the oracle the old code failed. Per that file's standing convention the assertions are on the
  call PyBNF constructs, never on dask internals or a pinned dask version; the resulting worker
  shape above was confirmed against a real cluster by hand.
* **The remote paths are untouched.** `scheduler_file` / `scheduler_node` connect to a cluster
  someone else configured, and `dask-ssh` already passed `--nthreads 1`.
* **Intra-worker threading that PyBNF controls is unaffected.** `net_model._run_ss_scan_threaded`
  still runs an unscored steady-state scan on its own thread pool, having prepared the point models
  sequentially first; that pool is scoped and audited, which is exactly what dask's worker threads
  were not.

## Alternatives considered

* **Make the explicit branch multi-threaded instead, resolving the disagreement the other way.**
  Rejected: it contradicts the measured race, ADR-0065, and PyBNF's own documented advice for
  manual dask setups, and would make the safe configuration unreachable.
* **Pin `n_workers=cpu_count()` explicitly alongside `threads_per_worker=1`.** Rejected: it makes
  PyBNF's arithmetic the authority on core count and ignores affinity masks and cgroup quotas that
  dask already reads. Delegating gives the same number on a workstation and a better one in a
  container or a cpuset-confined allocation.
* **Add a `worker_threads` key so the old behavior stays reachable.** Rejected as documented
  above: config surface whose only purpose is to re-enable a configuration the project treats as
  unsafe.
* **Leave the default alone and wait for the upstream `sympy_to_c` fix.** Rejected: that fix
  addresses one instance. The general claim -- a C++ engine and module-level codegen caches are not
  thread-safe -- is what the rest of the codebase already assumes, and a per-instance fix does not
  make the default configuration agree with it.
