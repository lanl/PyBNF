# PyBioNetFit (PyBNF)

PyBioNetFit (PyBNF) helps you do the *workflows beyond simulation* that real modeling projects need: curve fitting, bootstrapping, model checking, and Bayesian uncertainty quantification (UQ) via MCMC sampling—while staying in familiar modeling languages (BNGL, SBML) and taking advantage of parallel compute resources. It was designed for rule-based models but also supports traditionally formulated deterministic and stochastic models.

---

## Why PyBNF?

- **Workflow-first:** Treat fitting, model checking, and UQ as first-class citizens alongside simulation. Encode qualitative properties (e.g., “output crosses a threshold”) using **BPSL** and mix them with quantitative data in the same objective function.
- **Standards-based & reproducible:** Works with **BNGL** and **SBML** models, which encourages sharing and reproduction of results.
- **Parallel by default:** Distributes simulations over cores and clusters using **Dask Distributed**, so you can scale from a laptop to a cluster without writing parallel code.
- **Broad algorithm toolbox:** Robust metaheuristics for hard, nonconvex problems and **MCMC** methods for Bayesian inference; experiment with different algorithms and settings.
- **Best-in-class for rule-based models:** Especially strong BNGL support and notable support for stochastic simulation workflows.

---

## What can I use it for?

- Fit model parameters to **time-series** and **dose-response** data.
- Combine **qualitative** constraints (BPSL) with quantitative data in one objective.
- **Check** model consistency with BPSL-encoded system properties.
- Perform **bootstrapping** and **MCMC sampling** for uncertainty analysis.

---

## Installation

Requirements: Python 3 on Linux, macOS, or Windows; no root access is typically needed (great for shared clusters). Install from PyPI:

```bash
python -m pip install pybnf
```

**SBML** models run via **libRoadRunner** (installed automatically with `pybnf[sbml]` if needed).  
**BNGL** models run via **BioNetGen**, which you install separately and reference via the `bng_command` option in your CONF file or by setting `BNGPATH`.

**Optional (SBML support via libRoadRunner):**
```bash
python -m pip install "pybnf[sbml]"
```
**Install BioNetGen (BNGL) separately** — see the [BioNetGen project and README](https://github.com/RuleWorld/bionetgen#readme).

**Then set the BioNetGen path (bash/zsh):**
```bash
export BNGPATH=/path/to/BioNetGen-X.Y.Z
```

**Or declare it in your PyBNF `.conf` file:**
```ini
bng_command=/path/to/BioNetGen-X.Y.Z/BNG2.pl
```

**Windows (PowerShell):**
```powershell
$env:BNGPATH = "C:\Path\To\BioNetGen-X.Y.Z"
```

---

## Quick start

1. **Prepare your files**
   - Model: BNGL (`.bngl`) or SBML (`.xml`)
   - Data: one or more white space-delimited files (`.exp`), the same format as BioNetGen's GDAT files
   - (Optional) Properties: BPSL (`.prop`)
   - Config: a single **CONFIGURATION** file (`.conf`) that declares paths to model and data files, parameters, bounds, data-model mappings, objective function, and algorithmic parameter settings

2. **Run a fit**
   ```bash
   pybnf -c path/to/your.conf
   ```

3. **Inspect results**
   - Check `Results/` for best-fit parameters, trajectories, and logs
   - Use your algorithm’s settings to refine, bootstrap, or sample

*Tip:* Parallelism and cluster settings are controlled in the CONF file; PyBNF will schedule simulations via Dask automatically.

---

## Algorithms (high level)

PyBNF bundles multiple **metaheuristic optimizers** (e.g., Differential Evolution, Particle Swarm, Scatter Search; Simplex for refinement) for curve fitting/maximum likelihood estimation and **MCMC samplers** for Bayesian inference. Having several options encourages testing what works best for your model and data.

---

## How it fits with your tools

- **SBML:** Simulated via libRoadRunner.
- **BNGL:** Simulated via BioNetGen.
- **Parallel compute:** Uses `dask.distributed` under the hood; you usually don’t need to interact with Dask directly.

---

## Parallelism and clusters

PyBNF uses **Dask Distributed** under the hood. On a laptop/desktop, it runs a **local cluster** (one process per worker). On an HPC system, it can connect to a **scheduler** you launch via your batch system (e.g., SLURM) and farm out simulations to workers.

- **Local:** default uses up to *N–1* cores to keep the machine responsive. You can cap workers via `parallel_count` in your `.conf`.
- **SLURM/cluster:** specify a scheduler file or scheduler/worker nodes in your `.conf`; see examples in the docs and test configs. PyBNF does not require root privileges—just a shared filesystem and an existing Dask deployment.

---

## Interrupting and resuming runs

**TL;DR:** Use **Ctrl‑C** to stop; **do not** use Ctrl‑Z. Resume with `--resume`.

### How to stop a run safely
- **macOS & Linux:** Press **Ctrl‑C** in the terminal running `pybnf`.
- **Windows (cmd or PowerShell):** Press **Ctrl‑C** in the same window.
- **Do not use Ctrl‑Z (or job‑control equivalents).** That only *suspends* the foreground process and can leave worker processes running in the background.

> Accidentally hit Ctrl‑Z on macOS/Linux? Type `fg` to bring the job back to the foreground, then press **Ctrl‑C**.

### Resume from the last checkpoint
Use the **`--resume`** flag to continue where the last checkpoint left off:
```bash
pybnf -c path/to/my.conf --resume
```

If a previous run has fully finished and you want to **extend** it by more iterations, pass a number:
```bash
pybnf -c path/to/my.conf --resume 50
```
This adds 50 more iterations beyond the completed run.

> Note: `--resume` and `--overwrite` are mutually exclusive—use one or the other.

### What gets restored
PyBioNetFit periodically writes checkpoints (e.g., `alg_backup.bp`) in your `output_dir`. `--resume` picks up from the latest checkpoint automatically.

---

## Networking note (Wi‑Fi)

Some users with **slow or congested Wi‑Fi** have observed sluggish web browsing while a large local PyBNF run is active. This is due to many long‑lived TCP connections and periodic heartbeats between local Dask components, which can interact poorly with limited Wi‑Fi capacity (bufferbloat on some home routers).

**What to do:**
- **Temporarily disable Wi‑Fi** during the run.
- **Limit local workers** by setting `parallel_count` in your `.conf`. This reduces concurrent connections and background traffic.
- Close other network‑heavy apps (e.g., cloud sync).

> On **wired Ethernet**, this effect is rarely noticeable and should not impact other users on the network. On HPC clusters, traffic remains on the internal fabric; end‑user Wi‑Fi is not involved.

---

## Console noise

- **“Port 8787 is already in use”** — this is the Dask dashboard port. PyBNF disables the dashboard in local runs by default. If you see this warning, it’s harmless.
- **KeyboardInterrupt (Ctrl‑C) noise** — PyBNF traps and softens Dask worker shutdown messages; if you still see traces, they’re benign.

---

## Citing PyBioNetFit

If you use PyBNF in your work, please cite:

> Mitra, E.D., Suderman, R., Colvin, J., Ionkov, A., Hu, A., Sauro, H.M., Posner, R.G., Hlavacek, W.S. (2019). **PyBioNetFit and the Biological Property Specification Language.** *iScience* 19, 1012–1036.

---

## Contributing

Contributions are welcome! See `CONTRIBUTING.md` for guidelines. The architecture intentionally makes it straightforward to add new optimizers and samplers.

---

## License

See `LICENSE` in this repository.

LANL code designation: C18062

---

## A note on scope

PyBNF aims to *lower the barrier* to parameterization, especially for BNGL/SBML models and stochastic workflows. For some ODE problems, gradient-based methods may be more efficient; choose the tool that best fits your task.
