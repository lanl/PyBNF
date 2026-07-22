#!/usr/bin/env python
"""Reproduction figure for the egfr_simpull job (Salazar-Cavazos et al. 2020).

Simulates egfr_simpull.bngl at its shipped nominals -- which ARE the authors' PyBNF best fit
(190127_CHO_EGFR_best-fit.bngl) -- and overlays the model on the two SiMPull fit datasets:

  dose response (t_end = 300 s):  pY1068_percent(), pY1173_percent(), phosR_per() vs EGF dose
  time course (25 nM EGF):        the same three observables vs time (0-300 s)

The three observables are the % of EGFR phosphorylated as read by anti-pY1068, anti-pY1173, and
a pan-phosphotyrosine (PY) antibody. An independent PyBNF de+refine fit reaches chi_sq = 361.3,
essentially equal to these params' 359.7 -- i.e. the shipped nominals sit at the global optimum,
so this deterministic ODE reproduction IS the paper's fit.

The network is finite (75 species) and generates in ~1 s; it is dose- and kinetics-independent,
so this generates it ONCE and reuses it for the dense dose sweep + the time course.
Requires BNGPATH set (BNG2.pl) and matplotlib/numpy.  Usage: BNGPATH=... python make_reproduction.py
"""
import glob
import os
import subprocess
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BNG = os.path.join(os.environ["BNGPATH"], "BNG2.pl")
MODEL = os.path.join(HERE, "egfr_simpull.bngl")

OBS = ["pY1068_percent", "pY1173_percent", "phosR_per"]
TITLES = {"pY1068_percent": "pY1068 (%)", "pY1173_percent": "pY1173 (%)", "phosR_per": "pan-PY (%)"}
FIT_DOSES = np.array([0.5e-9, 5.0e-9, 50.0e-9])          # the 3 dose_resp.exp doses (M)
DENSE_DOSES = np.logspace(-10.6, -7.0, 24)               # smooth dose curve (M)


def _cols(gdat_path):
    with open(gdat_path) as fh:
        header = fh.readline().lstrip("#").split()
    data = np.loadtxt(gdat_path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], {name: data[:, i] for i, name in enumerate(header)}


def simulate_all():
    """One BNG2.pl run at the shipped best-fit nominals: generate the network once, then a
    dense EGF dose sweep (parameter_scan over EGFconc, each dose to t_end=300 s) + the 25 nM
    time course (0-300 s)."""
    with open(MODEL) as fh:
        src = fh.read().split("end model")[0] + "end model\n"
    doses = ",".join(f"{d:.10g}" for d in DENSE_DOSES)
    actions = [
        "begin actions",
        "generate_network({overwrite=>1})",
        'saveParameters("p0")', 'saveConcentrations("c0")',
        f'parameter_scan({{suffix=>"dose",parameter=>"EGFconc",par_scan_vals=>[{doses}],'
        f'method=>"ode",t_start=>0,t_end=>300,n_steps=>3,print_functions=>1}})',
        'resetParameters("p0")', 'resetConcentrations("c0")',
        'setParameter("EGFconc","25.0e-9")', 'setConcentration("EGF(EGFL)","EGF_total")',
        'simulate({suffix=>"tc",method=>"ode",t_start=>0,t_end=>300,n_steps=>300,print_functions=>1})',
        "end actions",
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "egfr_simpull.bngl")
        with open(path, "w") as fh:
            fh.write(src + "\n" + "\n".join(actions) + "\n")
        r = subprocess.run(["perl", BNG, path], cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"BNG2.pl failed (rc={r.returncode}):\n{r.stderr[-2000:]}")
        _, scan = _cols(glob.glob(os.path.join(d, "*_dose.scan"))[0])   # one row per dose @ t_end
        t_tc, tc = _cols(glob.glob(os.path.join(d, "*_tc.gdat"))[0])
    return scan, (t_tc, tc)


def _metrics(lab, xm, ym, xd, yd, sd):
    yi = np.interp(xd, xm, ym)
    ok = ~np.isnan(yd)
    chisq = 0.5 * float(np.sum(((yi[ok] - yd[ok]) / sd[ok]) ** 2))
    rel = np.abs(yi[ok] - yd[ok]) / yd[ok]
    print(f"  {lab:26s} chi_sq={chisq:7.2f}  median|relerr|={np.median(rel):.3f}  max={rel.max():.3f}")
    return chisq, list(rel)


def main():
    dr = np.genfromtxt(os.path.join(HERE, "dose_resp.exp"), names=True, deletechars="")
    tc_d = np.genfromtxt(os.path.join(HERE, "EGF_25nM.exp"), names=True, deletechars="")
    scan, (t_tc, tc) = simulate_all()

    print("egfr_simpull reproduction at the authors' PyBNF best fit (190127_CHO_EGFR_best-fit.bngl):")
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    total_chi = 0.0
    all_rel = []
    for c, o in enumerate(OBS):
        # top row: dose response
        a = ax[0, c]
        a.errorbar(dr["EGFconc"] * 1e9, dr[o + "()"], yerr=dr[o + "_SD"], fmt="o", color="#222",
                   capsize=3, zorder=5, label="SiMPull (data)")
        a.plot(DENSE_DOSES * 1e9, scan[o], "-", color="#1f77b4", lw=2, label="model @ best-fit")
        chi, rel = _metrics(f"dose  {o}", DENSE_DOSES * 1e9, scan[o],
                            dr["EGFconc"] * 1e9, dr[o + "()"], dr[o + "_SD"])
        total_chi += chi
        all_rel += rel
        a.set(xscale="log", title=f"dose response -- {TITLES[o]}", xlabel="EGF (nM)", ylabel=TITLES[o])
        # bottom row: time course
        b = ax[1, c]
        b.errorbar(tc_d["time"], tc_d[o + "()"], yerr=tc_d[o + "_SD"], fmt="o", color="#222",
                   capsize=3, zorder=5, label="SiMPull (data)")
        b.plot(t_tc, tc[o], "-", color="#d62728", lw=2, label="model @ best-fit")
        chi, rel = _metrics(f"time  {o}", t_tc, tc[o], tc_d["time"], tc_d[o + "()"], tc_d[o + "_SD"])
        total_chi += chi
        all_rel += rel
        b.set(title=f"time course (25 nM) -- {TITLES[o]}", xlabel="time (s)", ylabel=TITLES[o])
    for a in ax.ravel():
        a.legend(frameon=False, fontsize=8)
        a.grid(alpha=0.25)
    print(f"\nTOTAL chi_sq = {total_chi:.2f}   median |rel err| = {np.median(all_rel)*100:.1f}%  "
          f"(max {max(all_rel)*100:.1f}%)")
    fig.suptitle("Salazar-Cavazos 2020 -- multisite EGFR phosphorylation, CHO EGFR-GFP "
                 f"(PyBNF best-fit, chi_sq = {total_chi:.0f})", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "egfr_simpull_reproduction.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
