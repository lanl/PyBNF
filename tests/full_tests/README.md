# PyBNF Full Test Suite

This directory contains a comprehensive test suite for validating PyBNF functionality, including multi-machine cluster execution.

## Test Cases

The suite includes 7 test problems (T1-T7) that exercise different PyBNF features:
- **T1-ssprop**: Scatter search with polynomial fitting
- **T2-ade-abcd**: Asynchronous differential evolution with refinement
- **T3-de-egg**: Differential evolution with bootstrap
- **T4-pso-nf**: Particle swarm optimization
- **T5-pt-trivial**: Parallel tempering on trivial problem
- **T6-check**: Model checking (`fit_type = check`)
- **T7-dream-trivial**: DREAM algorithm on trivial problem

## Running the Tests

### Local Execution (Single Machine)

The simplest way to run the test suite:

```bash
python3 run_all.py
```

This runs all tests locally without requiring cluster configuration. Results are written to `test_summary.txt`.

**Expected runtime:** ~30 minutes on a typical workstation

### Cluster Execution (Multi-Machine)

Two SLURM batch scripts are provided for running on a cluster:

#### Option 1: Automatic Cluster Setup (`cluster.sh`)

Uses PyBNF's built-in SSH-based cluster management:

```bash
sbatch cluster.sh
```

**Before running:**
1. Edit the `PYTHON ENVIRONMENT` section to activate your PyBNF environment
2. Adjust resource configuration if needed (nodes, CPUs, time limit)

#### Option 2: Manual Dask Cluster (`cluster_manual.sh`)

Manually configures a Dask scheduler and workers:

```bash
sbatch cluster_manual.sh
```

**Before running:**
1. Edit the `PYTHON ENVIRONMENT` section to activate your PyBNF environment
2. Edit `WORKERS_PER_NODE` and `THREADS_PER_WORKER` to match your resources
3. Adjust resource configuration if needed (nodes, CPUs, time limit)

**When to use this:**
- When you need fine-grained control over Dask worker configuration
- When automatic SSH setup doesn't work on your cluster
- When debugging cluster connectivity issues

### Resource Requirements

**Default configuration:**
- **Nodes:** 2
- **CPUs per node:** 36
- **Time limit:** 1 hour
- **Total cores:** 72

These values can be modified at the top of each batch script.

## Interpreting Results

### Output Files

After running the test suite, check:

- **Summary file**: `run_all.py` writes `test_summary.txt` for a local run,
  `test_summary_ssh.txt` for the `ssh` mode, and `test_summary_sf.txt` for the
  `sf` mode.
- **`T*/fit/Results/`**: Detailed results for each test case

### Reference Output

Example outputs are provided for comparison:

- **`example_summary.txt`**: Single-machine run from March 2019 (v1.0.0 release)
- **`example_summary_ssh.txt`**: Multi-machine run using SSH mode
- **`example_summary_sf.txt`**: Multi-machine run using manual dask setup

**Note:** These reference files are historical. Your output will differ due to:
- Algorithm changes and improvements
- Different random number seeds
- Hardware differences (CPU speed, core count)
- Numerical precision variations

**What to check:**
- All tests should complete without errors
- Objective function values should be reasonable (similar order of magnitude)
- Test problems should converge (not diverge or fail)

## Troubleshooting

### Environment Issues

If you see `ModuleNotFoundError` or import errors:
- Check that the active environment has PyBNF installed. Run `uv pip list | grep pybnf`, or `pip list | grep pybnf` if you installed PyBNF with pip.
- Install the dependencies with `uv sync`. This project is managed with uv and a populated `uv.lock` file, and it has no `requirements.txt`. See `CONTRIBUTING.md` for more.
- Make sure the batch script activates the correct environment.

### Cluster Issues

If workers fail to connect:
- Check SSH connectivity between nodes: `ssh <node> hostname`
- Verify the scheduler file (`sf`) is on a shared filesystem
- Check firewall settings allow communication on Dask ports (8786, 8787)
- Review the SLURM job output file for error messages

### Dask Command Not Found

If you see `dask: command not found`:
- Check that the distributed package is installed. Run `uv pip list | grep distributed`.
- Make sure the environment is activated.
- This project needs the distributed package at version 2024.1.0 or newer, as set in `pyproject.toml`. The cluster commands are `dask scheduler` and `dask worker`, which are subcommands of the `dask` program. The older standalone programs named `dask-scheduler` and `dask-worker` stopped installing in distributed version 2026.6.0, as noted in `CHANGELOG.md`.

## Contributing

When modifying the multi-machine execution code in PyBNF:
1. Run the local test suite: `python3 run_all.py`
2. If you have cluster access, run one of the cluster tests
3. Compare results to verify no regressions
4. Document any new configuration requirements

## See Also

- [PyBNF Documentation](https://pybnf.readthedocs.io/)
- [Dask Distributed CLI Documentation](https://docs.dask.org/en/stable/deploying-cli.html)
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - General contribution guidelines
