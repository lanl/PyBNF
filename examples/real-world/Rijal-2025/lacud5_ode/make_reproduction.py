#!/usr/bin/env python
"""Reproduce the lacUD5 mean/Fano curve from the EXACT moment ODEs.

The script integrates the same in-process BNGsim model PyBNF uses during fitting, at both the
Rijal and Mehta (2025) Fig. 7 parameters and this job's own fitted optimum, and checks the
integrated moments against the closed-form telegraph solution.

Usage:
    BNGPATH=... python make_reproduction.py
"""

import os
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
CONF = HERE / "lacud5_ode.conf"
EXP = HERE / "lacud5.exp"
SOURCE = HERE / "jones_fig3a_source.tsv"
OUT = HERE / "lacud5_ode_reproduction.png"
SUFFIX = "lacud5"
LABEL = "lacUD5"

# Rijal and Mehta (2025), Fig. 7. Rates are normalized by koff_R=1.
PUBLISHED = {"r_over_gamma": 90.25 / 6.20, "gamma": 6.20}
# This job's converged gntr optimum (see VALIDATION.md).
FITTED = {"r_over_gamma": 13.608555430613906, "gamma": 8.15028716549539}


def _load_experimental_fano():
    raw = np.genfromtxt(SOURCE, comments="#",
                        dtype=[("promoter", "U16"), ("mean", float), ("fano", float)])
    _, first = np.unique(raw["mean"], return_index=True)
    selected = raw[np.sort(first)]
    return selected["mean"], selected["fano"]


def _build(output_dir: Path):
    raw = parse.ploop(CONF.read_text(encoding="utf-8").splitlines(keepends=True))
    raw["output_dir"] = str(output_dir)
    raw["verbosity"] = 0
    raw["population_size"] = 4
    raw["max_iterations"] = 1
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
    return model, cfg


def _integrate(model, cfg, values, work: Path, tag: str):
    pset = PSet([var.set_value(values[var.name]) for var in cfg.variables])
    result = model.copy_with_param_set(pset).execute(str(work), tag, timeout=300)[SUFFIX]
    mean = np.asarray(result["m_mean"], dtype=float)
    second = np.asarray(result["m_second"], dtype=float)
    return mean, np.sqrt(np.maximum(second - mean**2, 0.0))


def _closed_form(mu, values):
    """Steady-state telegraph moments: <m> = (r/gamma) p, Fano = 1 + r(1-p)/(kon+koff+gamma)."""
    capacity, gamma = values["r_over_gamma"], values["gamma"]
    kon = capacity / mu - 1.0
    r = capacity * gamma
    mean = capacity / (kon + 1.0)
    fano = 1.0 + r * kon / ((kon + 1.0) * (kon + 1.0 + gamma))
    return mean, np.sqrt(mean * fano), fano


def main() -> None:
    fit_data = np.loadtxt(EXP)
    mu, target_mean, target_sd = fit_data[:, 0], fit_data[:, 1], fit_data[:, 2]
    source_mean, target_fano = _load_experimental_fano()
    if not np.allclose(source_mean, target_mean):
        raise RuntimeError("The source-data and .exp mean columns have drifted")

    with tempfile.TemporaryDirectory(prefix="ode-twin-") as tmp:
        work = Path(tmp)
        model, cfg = _build(work / "output")
        mean_pub, sd_pub = _integrate(model, cfg, PUBLISHED, work, "published")
        mean_fit, sd_fit = _integrate(model, cfg, FITTED, work, "fitted")

    ex_mean_pub, ex_sd_pub, fano_pub = _closed_form(mu, PUBLISHED)
    ex_mean_fit, ex_sd_fit, fano_fit = _closed_form(mu, FITTED)

    # Rijal Eq. 14 = sum of squared mean error + squared SD error. NOTE: PyBNF's edition-2
    # `sos` desugars to Gaussian(sigma=1), whose kernel carries a factor 1/2, so
    #     Rijal Eq. 14 == 2 * (the objective PyBNF reports).
    def eq14(m, s):
        return float(np.sum((m - target_mean) ** 2 + (s - target_sd) ** 2))
    print(f"backend: BngsimModel, method=ode (exact moment equations), {LABEL}")
    print(f"published r/gamma={PUBLISHED['r_over_gamma']:.6g}, gamma={PUBLISHED['gamma']:.6g}")
    print(f"   Rijal Eq. 14 sos = {eq14(mean_pub, sd_pub):.6f}  "
          f"(PyBNF objective {eq14(mean_pub, sd_pub) / 2:.6f})")
    print(f"fitted    r/gamma={FITTED['r_over_gamma']:.6g}, gamma={FITTED['gamma']:.6g}")
    print(f"   Rijal Eq. 14 sos = {eq14(mean_fit, sd_fit):.6f}  "
          f"(PyBNF objective {eq14(mean_fit, sd_fit) / 2:.6f})")
    print("ODE vs closed form (max abs):")
    print(f"   mean  published {np.abs(mean_pub - ex_mean_pub).max():.3e}, "
          f"fitted {np.abs(mean_fit - ex_mean_fit).max():.3e}")
    print(f"   SD    published {np.abs(sd_pub - ex_sd_pub).max():.3e}, "
          f"fitted {np.abs(sd_fit - ex_sd_fit).max():.3e}")

    fano_ode_pub = np.divide(sd_pub**2, mean_pub, out=np.full_like(mean_pub, np.nan),
                             where=mean_pub > 0)
    fano_ode_fit = np.divide(sd_fit**2, mean_fit, out=np.full_like(mean_fit, np.nan),
                             where=mean_fit > 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(target_mean, target_fano, "s", color="#222222", label="Jones 2014 data")
    axes[0].plot(target_mean, fano_ode_pub, "-", color="#d62728",
                 label="moment ODE @ Rijal Fig. 7")
    axes[0].plot(target_mean, fano_ode_fit, "--", color="#1f77b4",
                 label="moment ODE @ fitted optimum")
    axes[0].set(xscale="log", xlabel="mean mRNA per gene copy", ylabel="Fano factor")
    axes[0].set_title(f"{LABEL}: variability curve")

    axes[1].plot(target_mean, target_sd, "s", color="#222222", label="Jones-derived SD")
    axes[1].plot(target_mean, sd_pub, "-", color="#d62728", label="moment ODE @ Rijal Fig. 7")
    axes[1].plot(target_mean, sd_fit, "--", color="#1f77b4",
                 label="moment ODE @ fitted optimum")
    axes[1].set(xscale="log", xlabel="mean mRNA per gene copy",
                ylabel="mRNA standard deviation")
    axes[1].set_title(f"Rijal Eq. 14 target (fitted sos={eq14(mean_fit, sd_fit):.3f})")

    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(frameon=False)
    fig.suptitle(f"Two-state promoter, exact moment ODEs — {LABEL} "
                 "(deterministic twin of the exact-SSA job)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
