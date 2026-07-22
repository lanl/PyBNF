#!/usr/bin/env python
"""Reproduction figure for the egfr_ode job (Kozer 2013 model; PyBNF fit, Mitra 2019).

Simulates the model at the PyBNF/BioNetFit BEST-FIT parameters and overlays the model on the
four Kozer 2013 preprocessed datasets (each scaled so its own measurement average is 1). The
best-fit is from the PyBNF paper's Data S1 (RuleHub Published/Mitra2019/10-egfr/fit_de,
objective chi_sq = 10.164) -- NOT the model's shipped nominals, which are the *Kozer 2013*
Table-1 fit and reproduce the raw (un-normalized) data, sitting ~100x below these
average-normalized targets. The four alpha*_pre scale factors are the well-determined part of
the (sloppy) fit; the kinetic constants are non-identifiable (algorithms disagree at equal
objective), so this uses one representative optimum.

  time course (LT = 30 nM):   pre2_time = alpha2_pre*Clusters   (cluster density, Fig. 3B)
                              pre4_time = alpha4_pre*pEGFR      (phospho-EGFR,   Fig. 3D)
  dose-response (t_end=1200): pre1_dose = alpha1_pre*Clusters   (cluster density, Fig. 2B)
                              pre3_dose = alpha3_pre*pEGFR      (phospho-EGFR,   Fig. 2D)

The network is finite only under max_stoich (kept in the .bngl) but still cluster-scale to
generate (> several min); it is topology-independent of the ligand dose and the fitted
kinetics, so this generates it ONCE and reuses it for the time course + dose sweep.
Requires BNGPATH set (BNG2.pl) and matplotlib/numpy.  Usage: BNGPATH=... python make_reproduction.py
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
MODEL = os.path.join(HERE, "egfr_ode.bngl")

# PyBNF best-fit, RuleHub Published/Mitra2019/10-egfr/fit_de (chi_sq = 10.164).
BESTFIT_KIN = dict(k_o=0.20131, k_c=0.38559, kaf=161.351, kar=5.5574, chi_r=1770.02)
A1, A2, A3, A4 = 25.6534, 29.0337, 39.7741, 30.5021   # alpha1..4_pre (best-fit scale factors)
DENSE_DOSES = np.logspace(-3, 2, 16)


def _cols(gdat_path):
    with open(gdat_path) as fh:
        header = fh.readline().lstrip("#").split()
    data = np.loadtxt(gdat_path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], {name: data[:, i] for i, name in enumerate(header)}


def simulate_all():
    """One BNG2.pl run at the best-fit kinetics: generate the (capped) network once, then
    the time course + a dose sweep (resetConcentrations + setConcentration('EGF(rec)'))."""
    with open(MODEL) as fh:
        src = fh.read().split("end model")[0] + "end model\n"
    for k, v in BESTFIT_KIN.items():
        src = re.sub(rf"(^{k}\s+)[\d.eE+-]+", rf"\g<1>{v:.10g}", src, count=1, flags=re.M)
    with tempfile.TemporaryDirectory() as d:
        actions = ["generate_network({overwrite=>1,max_stoich=>{EGF=>4,EGFR=>4}})",
                   "saveConcentrations()", 'setConcentration("EGF(rec)",30)',
                   'simulate({suffix=>"tc",method=>"ode",t_start=>0,t_end=>600,n_steps=>600})']
        for i, dose in enumerate(DENSE_DOSES):
            actions += ["resetConcentrations()", f'setConcentration("EGF(rec)",{dose:.10g})',
                        f'simulate({{suffix=>"d{i}",method=>"ode",t_start=>0,t_end=>1200,n_steps=>240}})']
        path = os.path.join(d, "egfr_ode.bngl")
        with open(path, "w") as fh:
            fh.write(src + "\n" + "\n".join(actions) + "\n")
        subprocess.run(["perl", BNG, path], cwd=d, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t_tc, tc = _cols(glob.glob(os.path.join(d, "*_tc.gdat"))[0])
        clus_ss, peg_ss = [], []
        for i in range(len(DENSE_DOSES)):
            _, dd = _cols(glob.glob(os.path.join(d, f"*_d{i}.gdat"))[0])
            clus_ss.append(dd["Clusters"][-1])
            peg_ss.append(dd["pEGFR"][-1])
    return (t_tc, tc["Clusters"], tc["pEGFR"]), (np.array(clus_ss), np.array(peg_ss))


def _metrics(lab, xm, ym, xd, yd, sd):
    yi = np.interp(xd, xm, ym)
    ok = ~np.isnan(yd)
    chisq = 0.5 * float(np.sum(((yi[ok] - yd[ok]) / sd[ok]) ** 2))
    rel = np.abs(yi[ok] - yd[ok]) / yd[ok]
    print(f"  {lab:22s} chi_sq={chisq:7.2f}  median|relerr|={np.median(rel):.3f}  max={rel.max():.3f}")


def main():
    tc = np.genfromtxt(os.path.join(HERE, "timecourse.exp"), names=True, deletechars="")
    dr = np.genfromtxt(os.path.join(HERE, "doseresponse.exp"), names=True, deletechars="")
    (t_tc, clus_t, peg_t), (clus_ss, peg_ss) = simulate_all()
    m_pre2, m_pre4 = A2 * clus_t, A4 * peg_t
    m_pre1, m_pre3 = A1 * clus_ss, A3 * peg_ss

    print("egfr_ode reproduction at the PyBNF best-fit (Mitra 2019 Data S1, 10-egfr/fit_de):")
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].errorbar(tc["time"] / 60, tc["pre2_time"], yerr=tc["pre2_time_SD"], fmt="o",
                      color="#222", capsize=3, zorder=5, label="Kozer Fig. 3B")
    ax[0, 0].plot(t_tc / 60, m_pre2, "-", color="#1f77b4", lw=2, label="PyBNF best-fit")
    _metrics("cluster dens (time)", t_tc, m_pre2, tc["time"], tc["pre2_time"], tc["pre2_time_SD"])
    ax[0, 0].set(title="EGFR cluster density vs. time (30 nM)", xlabel="time (min)", ylabel="pre2_time")
    ax[0, 1].errorbar(tc["time"] / 60, tc["pre4_time"], yerr=tc["pre4_time_SD"], fmt="o",
                      color="#222", capsize=3, zorder=5, label="Kozer Fig. 3D")
    ax[0, 1].plot(t_tc / 60, m_pre4, "-", color="#d62728", lw=2, label="PyBNF best-fit")
    _metrics("phospho-EGFR (time)", t_tc, m_pre4, tc["time"], tc["pre4_time"], tc["pre4_time_SD"])
    ax[0, 1].set(title="phospho-EGFR vs. time (30 nM)", xlabel="time (min)", ylabel="pre4_time")
    ax[1, 0].errorbar(dr["LT"], dr["pre1_dose"], yerr=dr["pre1_dose_SD"], fmt="o",
                      color="#222", capsize=3, zorder=5, label="Kozer Fig. 2B")
    ax[1, 0].plot(DENSE_DOSES, m_pre1, "-", color="#1f77b4", lw=2, label="PyBNF best-fit")
    _metrics("cluster dens (dose)", DENSE_DOSES, m_pre1, dr["LT"], dr["pre1_dose"], dr["pre1_dose_SD"])
    ax[1, 0].set(xscale="log", title="EGFR cluster density vs. EGF dose", xlabel="LT (nM)", ylabel="pre1_dose")
    ax[1, 1].errorbar(dr["LT"], dr["pre3_dose"], yerr=dr["pre3_dose_SD"], fmt="o",
                      color="#222", capsize=3, zorder=5, label="Kozer Fig. 2D")
    ax[1, 1].plot(DENSE_DOSES, m_pre3, "-", color="#d62728", lw=2, label="PyBNF best-fit")
    _metrics("phospho-EGFR (dose)", DENSE_DOSES, m_pre3, dr["LT"], dr["pre3_dose"], dr["pre3_dose_SD"])
    ax[1, 1].set(xscale="log", title="phospho-EGFR vs. EGF dose", xlabel="LT (nM)", ylabel="pre3_dose")
    for a in ax.ravel():
        a.legend(frameon=False, fontsize=8)
        a.grid(alpha=0.25)
    fig.suptitle("Kozer 2013 EGFR ODE -- PyBNF best-fit (Mitra 2019, chi_sq=10.16)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "egfr_ode_reproduction.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
