# PyBNF interactive notebooks

A small collection of Jupyter notebooks showing **PyBNF + bngsim used
interactively** — fitting, simulating, sampling, and importing PEtab problems from
inside a running kernel, with the results plotted inline. They complement the
script-and-`.conf` lessons in [`../tutorial/`](../tutorial/): same edition-2 surface,
different entry point.

| Notebook | What it shows |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Write a model, make data, run a **differential-evolution** fit, plot fit-vs-data. The 5-minute tour. |
| [`02_bngsim_simulation.ipynb`](02_bngsim_simulation.ipynb) | Simulate a model **forward with bngsim** (no fitting): time courses, a parameter sweep, steady state. |
| [`03_posterior_exploration.ipynb`](03_posterior_exploration.ipynb) | Sample a posterior with **HMC/NUTS**, export an **ArviZ `InferenceData`**, and read trace / pair / R-hat / ESS. |
| [`04_petab_in_a_notebook.ipynb`](04_petab_in_a_notebook.ipynb) | **Import a PEtab problem** with `pybnf.petab.import_job` and fit it interactively. |
| [`05_gradient_fitting_profiles.ipynb`](05_gradient_fitting_profiles.ipynb) | A **Data2Dynamics-style** workflow: **multi-start trust-region** least-squares using bngsim **forward sensitivities** for the Jacobian, then **profile-likelihood** identifiability — on the CC0 Becker EpoR model with synthetic data. |

## How PyBNF runs in a notebook

The command line, `pybnf -c fit.conf`, starts a Dask *distributed* cluster and
farms each simulation out to worker **processes** — the right design for a large
production fit. Inside a notebook you usually want the opposite: a **single-process,
synchronous** run whose Python objects stay live in the kernel.

[`pybnf_notebook.py`](pybnf_notebook.py) provides that. Its `run_fit(conf_text,
out_dir)` drives PyBNF's *real* algorithm classes to completion inline, using a
drop-in synchronous stand-in for the Dask client — the same in-process execution
path PyBNF's own test suite uses. So the notebooks exercise production code, not a
reimplementation of it. The rest of the helper is small conveniences:

- `build_algorithm` — build a fit without running it (to reach `alg.model_list`);
- `best_fit` — winning parameters as a dict; `simulate` — a model's trajectory at chosen parameters;
- `load_exp` — read a `.exp` into a DataFrame;
- `bngl_to_net` / `compile_bngl` / `bngsim_model` — get from a `.bngl` to a ready `bngsim.Model` for pure forward simulation.

For a production run, put the same `.conf` on disk and run `pybnf -c` — only the
executor changes.

## Running these yourself

The notebooks are committed **pre-executed** (with outputs), so you can read them on
GitHub without a kernel. To re-run them:

```bash
# from the repo root, with the project's virtualenv active
export BNGPATH="$HOME/Simulations/BioNetGen-2.9.3"   # your BioNetGen install
jupyter lab examples/notebooks/          # …or execute headless:
jupyter nbconvert --to notebook --execute --inplace examples/notebooks/01_quickstart.ipynb
```

Dependencies (all in the project's `.venv`): `pybnf`, `bngsim` + `BNG2.pl`
(notebooks 01/02/04/05); the optional `jax` and `arviz` extras (notebook 03);
`petab` (notebook 04); `scipy` (notebook 05). Notebook 02/03/04's diagnostics also
use `pandas`, `matplotlib`, and (for the `.nc` export) `h5py`.

## CI

`tests/test_notebooks.py` re-executes each notebook top-to-bottom on the current
kernel as a **smoke test** (opt-in, `-m slow`; the bngsim notebooks are gated on
`BNGPATH`, the posterior notebook on the `jax`/`arviz` extras). It runs an in-memory
copy and does **not** overwrite the committed, pre-run notebooks. Regenerate the
committed outputs by executing them with `nbconvert --execute --inplace` (above).
