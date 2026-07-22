#!/usr/bin/env python
"""Reproduction figure for the egfr_nf job (Kozer et al. 2013, PMC3698845) -- NETWORK-FREE.

Simulates the model NETWORK-FREE with NFsim at its published parameters (the .bngl nominals),
under the two experimental designs, and overlays the model on the four Kozer datasets. NFsim
is stochastic, so each condition is averaged over several replicate runs (as the conf's
`smoothing` does during a fit). Observables are computed from the oligomer-species columns
(robust to NFsim function output):

  Clusters = sum(monomer..icosadecamer)               (20 EGFR oligomer sizes)
  pre2_time = alpha2_pre*Clusters/f  (cluster density, Fig. 3B);  pre4_time = alpha4_pre*pEGFR/f (Fig. 3D)
  pre1_dose = alpha1_pre*Clusters/f  (cluster density, Fig. 2B);  pre3_dose = alpha3_pre*pEGFR/f (Fig. 2D)

NFsim on ~10^3-10^4 molecules is cluster-scale; high ligand doses (LT_nM=100 -> ~10^6 EGF
particles) are the slow ones. Requires BNGPATH set (uses BNG2.pl -> NFsim) and matplotlib/numpy.
Usage:  BNGPATH=... python make_reproduction.py [n_time_reps n_dose_reps]
"""
import glob
import os
import re
import subprocess
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BNG = os.path.join(os.environ["BNGPATH"], "BNG2.pl")
MODEL = os.path.join(HERE, "egfr_nf.bngl")

# PyBNF best-fit, RuleHub Published/Mitra2019/04-egfrnf/fit_ade (chi_sq = 13.405). The shipped
# .bngl nominals are GenFit STARTING values (example2_starting_point.bngl), ~100x below the
# average-normalized data; these fitted alpha*_pre scale factors are what the data pins down.
BESTFIT_KIN = dict(k_o=3.54931, k_c=15.8220, kaf=45.0460, kar=0.64255, chi_r=1907.32)
A1, A2, A3, A4, F = 2.5718e-5, 2.7509e-5, 5.0739e-5, 3.2791e-5, 0.01
OLIGO = ("monomer dimer trimer tetramer pentamer hexamer heptamer octamer nonamer decamer "
         "undecamer dodecamer tridecamer tetradecamer pentadecamer hexadecamer heptadecamer "
         "octadecamer nonadecamer icosadecamer").split()
DATA_DOSES = [0.001, 0.1, 1.0, 10.0, 100.0]   # doseresponse.exp doses (0.01 nM is NaN)
N_TIME = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_DOSE = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def _cols(gdat_path):
    with open(gdat_path) as fh:
        header = fh.readline().lstrip("#").split()
    data = np.loadtxt(gdat_path)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], {name: data[:, i] for i, name in enumerate(header)}


def _clusters_pegfr(cols):
    clusters = sum(cols[name] for name in OLIGO)
    return clusters, cols["pEGFR"]


def run_nf(lt_nm, t_end, n_steps, seed):
    """One network-free NFsim run at ligand dose lt_nm; returns (t, Clusters, pEGFR)."""
    with open(MODEL) as fh:
        src = fh.read().split("end model")[0] + "end model\n"
    src = re.sub(r"(^LT_nM\s+)[\d.eE+-]+", rf"\g<1>{lt_nm:.10g}", src, count=1, flags=re.M)
    for k, v in BESTFIT_KIN.items():   # override the fitted kinetics (starting-point -> best-fit)
        src = re.sub(rf"(^{k}\s+)[\d.eE+-]+", rf"\g<1>{v:.10g}", src, count=1, flags=re.M)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "egfr_nf.bngl")
        with open(path, "w") as fh:
            fh.write(src)
            fh.write(f'\nsimulate({{method=>"nf",t_end=>{t_end},n_steps=>{n_steps},'
                     f'seed=>{seed},gml=>1000000,complex=>1}})\n')
        subprocess.run(["perl", BNG, path], cwd=d, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t, cols = _cols(glob.glob(os.path.join(d, "*.gdat"))[0])
    clusters, pegfr = _clusters_pegfr(cols)
    return t, clusters, pegfr


def avg_runs(lt_nm, t_end, n_steps, n_reps):
    cl, pe, tref = [], [], None
    for r in range(n_reps):
        t, c, p = run_nf(lt_nm, t_end, n_steps, seed=1000 + r)
        tref = t if tref is None else tref
        cl.append(np.interp(tref, t, c))
        pe.append(np.interp(tref, t, p))
    return tref, np.mean(cl, axis=0), np.mean(pe, axis=0)


def _metrics(lab, xm, ym, xd, yd, sd):
    yi = np.interp(xd, xm, ym)
    ok = ~np.isnan(yd)
    chisq = 0.5 * float(np.sum(((yi[ok] - yd[ok]) / sd[ok]) ** 2))
    rel = np.abs(yi[ok] - yd[ok]) / yd[ok]
    print(f"  {lab:22s} chi_sq={chisq:8.2f}  median|relerr|={np.median(rel):.3f}  max={rel.max():.3f}")


def main():
    tc = np.genfromtxt(os.path.join(HERE, "timecourse.exp"), names=True, deletechars="")
    dr = np.genfromtxt(os.path.join(HERE, "doseresponse.exp"), names=True, deletechars="")
    print(f"egfr_nf NFsim reproduction at the PyBNF best-fit (Mitra 2019, 04-egfrnf/fit_ade) "
          f"(time reps={N_TIME}, dose reps={N_DOSE}):")

    t_tc, clus_t, peg_t = avg_runs(30.0, 600, 600, N_TIME)
    m_pre2, m_pre4 = A2 * clus_t / F, A4 * peg_t / F

    clus_ss, peg_ss = [], []
    for dose in DATA_DOSES:
        _, c, p = avg_runs(dose, 600, 60, N_DOSE)
        clus_ss.append(c[-1])
        peg_ss.append(p[-1])
    m_pre1 = A1 * np.array(clus_ss) / F
    m_pre3 = A3 * np.array(peg_ss) / F

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].errorbar(tc["time"] / 60, tc["pre2_time"], yerr=tc["pre2_time_SD"], fmt="o",
                      color="#222", capsize=3, zorder=5, label="Kozer Fig. 3B (mean +/- SD)")
    ax[0, 0].plot(t_tc / 60, m_pre2, "-", color="#1f77b4", lw=2, label="NFsim (published params)")
    _metrics("cluster dens (time)", t_tc, m_pre2, tc["time"], tc["pre2_time"], tc["pre2_time_SD"])
    ax[0, 0].set(title="EGFR cluster density vs. time (30 nM EGF)",
                 xlabel="time (min)", ylabel="pre2_time (scaled cluster density)")
    ax[0, 1].errorbar(tc["time"] / 60, tc["pre4_time"], yerr=tc["pre4_time_SD"], fmt="o",
                      color="#222", capsize=3, zorder=5, label="Kozer Fig. 3D (mean +/- SD)")
    ax[0, 1].plot(t_tc / 60, m_pre4, "-", color="#d62728", lw=2, label="NFsim (published params)")
    _metrics("phospho-EGFR (time)", t_tc, m_pre4, tc["time"], tc["pre4_time"], tc["pre4_time_SD"])
    ax[0, 1].set(title="phospho-EGFR vs. time (30 nM EGF)",
                 xlabel="time (min)", ylabel="pre4_time (scaled pEGFR)")
    ax[1, 0].errorbar(dr["LT_nM"], dr["pre1_dose"], yerr=dr["pre1_dose_SD"], fmt="o",
                      color="#222", capsize=3, zorder=5, label="Kozer Fig. 2B (mean +/- SD)")
    ax[1, 0].plot(DATA_DOSES, m_pre1, "o-", color="#1f77b4", lw=2, label="NFsim (published params)")
    _metrics("cluster dens (dose)", np.array(DATA_DOSES), m_pre1, dr["LT_nM"], dr["pre1_dose"], dr["pre1_dose_SD"])
    ax[1, 0].set(xscale="log", title="EGFR cluster density vs. EGF dose",
                 xlabel="EGF dose LT_nM (nM)", ylabel="pre1_dose (scaled cluster density)")
    ax[1, 1].errorbar(dr["LT_nM"], dr["pre3_dose"], yerr=dr["pre3_dose_SD"], fmt="o",
                      color="#222", capsize=3, zorder=5, label="Kozer Fig. 2D (mean +/- SD)")
    ax[1, 1].plot(DATA_DOSES, m_pre3, "o-", color="#d62728", lw=2, label="NFsim (published params)")
    _metrics("phospho-EGFR (dose)", np.array(DATA_DOSES), m_pre3, dr["LT_nM"], dr["pre3_dose"], dr["pre3_dose_SD"])
    ax[1, 1].set(xscale="log", title="phospho-EGFR vs. EGF dose",
                 xlabel="EGF dose LT_nM (nM)", ylabel="pre3_dose (scaled pEGFR)")
    for a in ax.ravel():
        a.legend(frameon=False, fontsize=8)
        a.grid(alpha=0.25)
    fig.suptitle("Kozer EGFR network-free (NFsim) -- PyBNF best-fit (Mitra 2019, chi_sq=13.41)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "egfr_nf_reproduction.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
