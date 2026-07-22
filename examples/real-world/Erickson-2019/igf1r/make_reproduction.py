#!/usr/bin/env python
"""Reproduction figure for the igf1r job -- Erickson 2019 Fig 3 at the paper's Table-1 params.

Runs the authors' own multi-phase protocol (the actions block in igf1r_legacy.bngl) through BNG2.pl with
the seven free rate constants set to Erickson 2019 Table 1 ("This Study"), then overlays the model
on the three fit datasets (Kiselyov 2009 Fig 5B/5D):

  * Panel A (F5B, steady-state competition, 4 h at 7 pM hot): model IGF1_hot_bound normalized to
    its no-competitor (first / lowest-cold scan row) value -- this is PyBNF's normalization=init.
  * Panel B (F5D_20min / F5D_60min, dissociation: 2 h preincubation at 24 pM hot, wash, then cold
    competition): model IGF1_hot_bound at 20 / 60 min normalized to B0 = the pre-wash bound amount
    (hot_bound at the end of the incubation = start of the dissociation phase). This "% remaining"
    normalization is what Erickson Fig 3B plots (the curves start at ~0.98, not pinned to 1.0);
    it differs from PyBNF's fit-time normalization=init (divide by first scan row) by a small
    per-experiment uniform factor (B0/row0 ~ 1.02 for 20 min, ~1.07 for 60 min).

The eighth constant a2prime is computed inside the model by the detailed-balance constraint.
Requires BNGPATH set (uses BNG2.pl) and matplotlib/numpy.
Usage:  BNGPATH=... python make_reproduction.py
"""
import glob
import os
import re
import subprocess
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BNG = os.path.join(os.environ["BNGPATH"], "BNG2.pl")
MODEL = os.path.join(HERE, "igf1r_legacy.bngl")

# Erickson 2019 Table 1 ("This Study") best-fit rate constants -> the model's __FREE tokens.
#   a1prime = kcr ; a2prime is derived in-model by detailed balance (Table 1 reports a2prime=52).
TABLE1 = {
    "a1_perMpers": 2.8e5,   # a1  (M^-1 s^-1)
    "d1":          5.0e-2,  # d1  (s^-1)
    "a2_perMpers": 1.5e4,   # a2  (M^-1 s^-1)
    "d2":          1.9e-4,  # d2  (s^-1)
    "kcr":         5.6e-3,  # a'1 (s^-1)
    "d1prime":     1.9e-5,  # d'1 (s^-1)
    "d2prime":     1.3e-2,  # d'2 (s^-1)
}


def run_protocol():
    """Set the model to Table-1 params, run its full actions block, return the raw outputs.

    Returns (scans, B0) where scans maps suffix -> (cold_doses, hot_bound) and B0 is the pre-wash
    bound hot (hot_bound at the end of the 2 h incubation = start of the dissociation phase).
    """
    with open(MODEL) as fh:
        src = fh.read()
    # Substitute the seven free tokens (declared as `name  name__FREE`) with the Table-1 values.
    for name, val in TABLE1.items():
        src = re.sub(rf"(?m)^{re.escape(name)}\s+{re.escape(name)}__FREE\b",
                     f"{name} {val:.10g}", src)
    params_block = re.search(r"begin parameters(.*?)end parameters", src, re.S).group(1)
    assert "__FREE" not in params_block, "unsubstituted __FREE token remains in parameters block"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "igf1r.bngl")
        with open(path, "w") as fh:
            fh.write(src)
        subprocess.run(["perl", BNG, path], cwd=d, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        scans = {}
        for suf in ("F5B", "F5D_20min", "F5D_60min"):
            arr = np.loadtxt(glob.glob(os.path.join(d, f"*_{suf}.scan"))[0])
            scans[suf] = (arr[:, 0], arr[:, 1])
        inc = np.loadtxt(glob.glob(os.path.join(d, "*_incubate.gdat"))[0])
        B0 = float(inc[-1, 1])   # hot_bound at t_end of the 2 h incubation
    return scans, B0


def matched(scan_dose, scan_val, exp_dose):
    """Model value at each experimental dose (nearest scan dose in log space)."""
    return np.array([scan_val[int(np.argmin(np.abs(np.log(scan_dose) - np.log(d))))]
                     for d in exp_dose])


def metrics(name, ym, ey, esd):
    chisq = 0.5 * float(np.sum(((ym - ey) / esd) ** 2))
    big = ey > 0.05
    rel = np.abs(ym[big] - ey[big]) / ey[big]
    print(f"  {name:10s} chi_sq(0.5*)={chisq:6.3f}  median|rel|={np.median(rel):.3f}  "
          f"max|rel|={rel.max():.3f}  (n={len(ey)}, y>0.05: {big.sum()})")
    return chisq


def main():
    scans, B0 = run_protocol()
    exp = {s: np.loadtxt(os.path.join(HERE, f"{s}.exp")) for s in ("F5B", "F5D_20min", "F5D_60min")}

    print("igf1r reproduction at Erickson 2019 Table 1 (BNG2.pl, full protocol):")
    print(f"  B0 (pre-wash bound hot) = {B0:.1f} copies")

    # Panel A: F5B, normalize to the first (lowest-cold) scan row  (== normalization=init).
    d, y, sd = exp["F5B"][:, 0], exp["F5B"][:, 1], exp["F5B"][:, 2]
    sc_d, sc_v = scans["F5B"]
    raw = matched(sc_d, sc_v, d)
    yA = raw / raw[0]
    cA = metrics("F5B", yA, y, sd)

    # Panel B: F5D_20/60, normalize to B0 ("% remaining", as Erickson Fig 3B plots).
    B = {}
    for suf in ("F5D_20min", "F5D_60min"):
        d2, y2, sd2 = exp[suf][:, 0], exp[suf][:, 1], exp[suf][:, 2]
        sc_d2, sc_v2 = scans[suf]
        B[suf] = (d2, matched(sc_d2, sc_v2, d2) / B0, y2, sd2)
    cB20 = metrics("F5D_20min", B["F5D_20min"][1], B["F5D_20min"][2], B["F5D_20min"][3])
    cB60 = metrics("F5D_60min", B["F5D_60min"][1], B["F5D_60min"][2], B["F5D_60min"][3])
    print(f"  TOTAL chi_sq (0.5*sum over 3 datasets) = {cA + cB20 + cB60:.3f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    oa = np.argsort(d)
    axA.errorbar(d[oa], y[oa], yerr=sd[oa], fmt="o", color="#222", ms=6, capsize=3, zorder=5,
                 label="Kiselyov 2009 Fig 5B (mean +/- SD)")
    axA.plot(d[oa], yA[oa], "s-", color="#1f77b4", lw=2, ms=5, label="model @ Erickson Table 1")
    axA.set(xscale="log", xlabel="unlabeled IGF1 concentration (M)",
            ylabel="relative level of bound hot IGF1",
            title="A  Competition, steady state (F5B)")
    axA.legend(frameon=False, fontsize=9)
    axA.grid(alpha=0.25, which="both")

    for suf, col, lab in (("F5D_20min", "#222", "20 min"), ("F5D_60min", "#17becf", "60 min")):
        dd, ym, yy, ss = B[suf]
        ob = np.argsort(dd)
        axB.errorbar(dd[ob], yy[ob], yerr=ss[ob], fmt="o", color=col, ms=6, capsize=3, zorder=5,
                     label=f"data {lab} (Fig 5D)")
        axB.plot(dd[ob], ym[ob], "s-", color=col, lw=2, ms=5, label=f"model {lab}")
    axB.set(xscale="log", xlabel="unlabeled IGF1 concentration (M)",
            ylabel="relative level of bound hot IGF1 (% remaining)",
            title="B  Dissociation after 2 h preincubation (F5D)")
    axB.legend(frameon=False, fontsize=9)
    axB.grid(alpha=0.25, which="both")

    fig.suptitle("IGF1 / IGF1R binding -- Erickson 2019 fit reproduced at the paper's Table-1 "
                 "parameters (reproduces Fig 3A/3B)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(HERE, "igf1r_reproduction.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
