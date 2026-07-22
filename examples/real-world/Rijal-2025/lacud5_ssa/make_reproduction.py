#!/usr/bin/env python
"""Reproduce the lacUD5 mean/Fano curve with exact BNGsim SSA trajectories.

The script evaluates the edition-2 job at the Rijal and Mehta (2025) Fig. 7 parameters. It uses
the same in-process BNGsim model that PyBNF uses during fitting, runs independent exact-SSA
parameter scans, and estimates E[m], E[m^2], SD, and the Fano factor across trajectories.

Usage:
    BNGPATH=... python make_reproduction.py [n_trajectories]
"""

import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pybnf import config, parse
from pybnf.pset import PSet
from pybnf.registry import FIT_TYPE_REGISTRY

HERE = Path(__file__).resolve().parent
CONF = HERE / "lacud5_ssa.conf"
EXP = HERE / "lacud5.exp"
SOURCE = HERE / "jones_fig3a_source.tsv"
OUT = HERE / "lacud5_ssa_reproduction.png"
SUFFIX = "lacud5"
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

# Rijal and Mehta (2025), Fig. 7. Rates are normalized by koff_R=1.
PUBLISHED = {"r_over_gamma": 90.25 / 6.20, "gamma": 6.20}


def _load_experimental_fano() -> tuple[np.ndarray, np.ndarray]:
    raw = np.genfromtxt(
        SOURCE,
        comments="#",
        dtype=[("promoter", "U16"), ("mean", float), ("fano", float)],
    )
    _, first = np.unique(raw["mean"], return_index=True)
    selected = raw[np.sort(first)]
    return selected["mean"], selected["fano"]


def _build_bngsim_model(output_dir: Path):
    raw = parse.ploop(CONF.read_text(encoding="utf-8").splitlines(keepends=True))
    raw["output_dir"] = str(output_dir)
    raw["verbosity"] = 0
    raw["population_size"] = 4
    raw["max_iterations"] = 1
    raw["smoothing"] = 1
    home = Path.cwd()
    os.chdir(HERE)
    try:
        cfg = config.Configuration(raw)
    finally:
        os.chdir(home)
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithm = FIT_TYPE_REGISTRY[cfg.config["fit_type"]].cls(cfg)
    model = algorithm.model_list[0]
    if type(model).__name__ != "BngsimModel":
        raise RuntimeError(f"Expected the BNGsim net backend, got {type(model).__name__}")
    pset = PSet([var.set_value(PUBLISHED[var.name]) for var in cfg.variables])
    return model, pset


def _ssa_moments(model, pset, work: Path) -> tuple[np.ndarray, np.ndarray]:
    samples = []
    for replicate in range(N_REPS):
        run_model = model.copy_with_param_set(pset)
        run_model._pybnf_replicate_index = replicate
        run_model._pybnf_stochastic_seed_policy = "auto"
        result = run_model.execute(str(work), f"replicate_{replicate}", timeout=120)
        samples.append(np.asarray(result[SUFFIX]["mRNA_count"], dtype=float))
    sample = np.vstack(samples)
    return sample.mean(axis=0), sample.var(axis=0)


def main() -> None:
    fit_data = np.loadtxt(EXP)
    target_mean, target_sd = fit_data[:, 1], fit_data[:, 2]
    source_mean, target_fano = _load_experimental_fano()
    if not np.allclose(source_mean, target_mean):
        raise RuntimeError("The source-data and .exp mean columns have drifted")

    with tempfile.TemporaryDirectory(prefix="lacud5-bngsim-") as tmp:
        work = Path(tmp)
        model, pset = _build_bngsim_model(work / "output")
        mean_ssa, var_ssa = _ssa_moments(model, pset, work)

    sd_ssa = np.sqrt(np.maximum(var_ssa, 0.0))
    fano_ssa = np.divide(var_ssa, mean_ssa, out=np.full_like(var_ssa, np.nan), where=mean_ssa > 0)

    capacity = PUBLISHED["r_over_gamma"]
    gamma = PUBLISHED["gamma"]
    kon = capacity / target_mean - 1.0
    r = capacity * gamma
    mean_exact = capacity / (kon + 1.0)
    fano_exact = 1.0 + r * kon / ((kon + 1.0) * (kon + 1.0 + gamma))
    sd_exact = np.sqrt(mean_exact * fano_exact)

    sos_ssa = float(np.sum((mean_ssa - target_mean) ** 2 + (sd_ssa - target_sd) ** 2))
    fano_mape = float(np.mean(np.abs(fano_ssa - target_fano) / target_fano))
    mean_rel = np.abs(mean_ssa - mean_exact) / np.maximum(mean_exact, 1e-12)
    sd_rel = np.abs(sd_ssa - sd_exact) / np.maximum(sd_exact, 1e-12)
    print(f"backend: BngsimModel, method=ssa, trajectories={N_REPS}")
    print(f"published: r={r:.4g}, gamma={gamma:.4g}, r/gamma={capacity:.6g}, koff_R=1")
    print(f"Rijal Eq. 14 sos (exact SSA moments): {sos_ssa:.6g}")
    print(f"mean Fano percentage error vs. data: {100 * fano_mape:.2f}%")
    print(f"SSA vs. analytical mean: median={np.median(mean_rel):.3g}, max={mean_rel.max():.3g}")
    print(f"SSA vs. analytical SD: median={np.median(sd_rel):.3g}, max={sd_rel.max():.3g}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(target_mean, target_fano, "s", color="#222222", label="Jones 2014 data")
    axes[0].plot(target_mean, fano_exact, "-", color="#d62728", label="exact moments")
    axes[0].plot(target_mean, fano_ssa, "o", mfc="none", color="#1f77b4", label="BNGsim SSA")
    axes[0].set(xscale="log", xlabel="mean mRNA per gene copy", ylabel="Fano factor")
    axes[0].set_title("lacUD5: variability curve")

    axes[1].plot(target_mean, target_sd, "s", color="#222222", label="Jones-derived SD")
    axes[1].plot(target_mean, sd_exact, "-", color="#d62728", label="exact moments")
    axes[1].plot(target_mean, sd_ssa, "o", mfc="none", color="#1f77b4", label="BNGsim SSA")
    axes[1].set(xscale="log", xlabel="mean mRNA per gene copy", ylabel="mRNA standard deviation")
    axes[1].set_title(f"Rijal Eq. 14 target (SSA sos={sos_ssa:.2f})")

    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(frameon=False)
    fig.suptitle("Two-state promoter at Rijal and Mehta (2025) Fig. 7 parameters")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
