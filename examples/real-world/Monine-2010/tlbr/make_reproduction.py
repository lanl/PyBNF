#!/usr/bin/env python
"""Reproduction figure for the tlbr job (Monine et al. 2010 model AND binding data).

Simulates the trivalent-ligand / bivalent-receptor model NETWORK-FREE with NFsim at Monine 2010's
OWN reported best-fit (Table 1, TLBR model), across the 12-dose LTconc titration, and overlays the
model FL (bound-ligand fraction) on the Monine 2010 Fig 2a binding data. The whole titration is run
with one BNGL `parameter_scan` per
replicate (exactly as the classic model's actions block did); NFsim is stochastic, so the doses
are averaged over several replicate scans (as the conf's `smoothing` does during a fit). The
readout is computed from the raw species columns (robust to NFsim function output):

  lambda = (L_total - L_free) / (2 * R_total)      FL = lambda / alpha

Each dose is integrated to t_end = 5000 s (the horizon the classic run used to reach steady state;
NFsim has no steady-state solve) and its final value is taken. Requires BNGPATH set (uses BNG2.pl
-> NFsim) and matplotlib/numpy.
Usage:  BNGPATH=... python make_reproduction.py [n_reps]
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
MODEL = os.path.join(HERE, "tlbr.bngl")

# Monine 2010 Table 1 best-fit for the TLBR model (the paper's OWN reported result):
#   K1 = 0.467 /nM (CI 0.111-0.767), K2 = 87.03 /nM (CI 31.6-128.1), alpha = 0.816 (CI 0.758-0.881).
# These ARE the shipped .bngl nominals. (A separate PyBNF/Mitra-2019 re-fit -- RuleHub
# Published/Mitra2019/11-TLBR/fit_ss init7, alpha=0.746, K1=0.109, K2=33.6, sos=0.00214 -- lands at
# a DIFFERENT point of the same sloppy K1-K2 valley; it reproduces the normalized FL1 but not the
# paper's lambda curve. See VALIDATION.md.)
BESTFIT = dict(alpha=0.816, K1=0.467, K2=87.03)
ALPHA = BESTFIT["alpha"]
T_END = 5000
N_STEPS = 10
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def _rows(scan_path):
    with open(scan_path) as fh:
        header = fh.readline().lstrip("#").split()
    data = np.loadtxt(scan_path)
    return {name: data[:, i] for i, name in enumerate(header)}


def scan_fl(doses, seed):
    """One network-free NFsim parameter_scan over all doses; returns final FL per dose."""
    with open(MODEL) as fh:
        src = fh.read().split("end model")[0] + "end model\n"
    for k, v in BESTFIT.items():   # set the fitted params to Monine's Table-1 best-fit
        src = re.sub(rf"(^{k}\s+)[\d.eE+-]+", rf"\g<1>{v:.12g}", src, count=1, flags=re.M)
    vals = ",".join(f"{d:.10g}" for d in doses)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tlbr.bngl")
        with open(path, "w") as fh:
            fh.write(src)
            fh.write(f'\nparameter_scan({{parameter=>"LTconc",par_scan_vals=>[{vals}],'
                     f'method=>"nf",complex=>1,gml=>10000000,print_functions=>1,'
                     f't_start=>0,t_end=>{T_END},n_steps=>{N_STEPS},suffix=>"scan",'
                     f'seed=>{seed},steady_state=>0,get_final_state=>0}})\n')
        subprocess.run(["perl", BNG, path], cwd=d, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cols = _rows(glob.glob(os.path.join(d, "*.scan"))[0])
    lam = (cols["L_total"] - cols["L_free"]) / (2.0 * cols["R_total"])
    return lam / ALPHA


def main():
    # tlbr.exp header is `# LTconc FL()`; the FL() column name has parens, so read positionally.
    exp = np.loadtxt(os.path.join(HERE, "tlbr.exp"))
    doses = exp[:, 0]
    fl_data = exp[:, 1]
    print(f"tlbr NFsim reproduction at Monine 2010 Table 1 (TLBR: K1=0.467, K2=87.03, alpha=0.816) "
          f"(reps={N_REPS}):")

    fl_model = np.mean([scan_fl(doses, seed=1000 + r) for r in range(N_REPS)], axis=0)

    # sos is the fit objective (data carry no _SD). Report relative error only where the data are
    # meaningfully above the near-zero baseline (the 3 lowest doses sit at the noise floor, one
    # even negative -- relative error there is meaningless).
    sos = float(np.sum((fl_model - fl_data) ** 2))
    big = np.abs(fl_data) > 0.05
    rel = np.abs(fl_model[big] - fl_data[big]) / np.abs(fl_data[big])
    # Monine 2010 fit the same data with RMS (SI Eq 11) and kept fits with RMS<0.02 (lambda space).
    rmslam = float(np.sqrt(np.mean((0.816 * fl_model - 0.816 * fl_data) ** 2)))
    print(f"  sos (fit objective, FL)    = {sos:.5f}")
    print(f"  RMS (lambda space)         = {rmslam:.4f}   (Monine kept fits with RMS<0.02, SI Eq 11)")
    print(f"  median |rel err| (FL>0.05) = {np.median(rel):.3f}")
    print(f"  max    |rel err| (FL>0.05) = {rel.max():.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(doses, fl_data, "o", color="#222", ms=8, zorder=5,
            label="Monine 2010 Fig 2a data (Biophys J 98:48)")
    ax.plot(doses, fl_model, "s-", color="#1f77b4", lw=2, ms=6,
            label="NFsim @ Monine Table 1 (sos=%.4f)" % sos)
    ax.set(xscale="log", xlabel="total ligand LTconc (nM)",
           ylabel="FL  (bound-ligand fraction = lambda/alpha)",
           title="Trivalent ligand / bivalent receptor -- network-free (NFsim)\n"
                 "Monine 2010 model + Fig 2a data at Monine's Table-1 fit (PyBNF corpus 11-TLBR)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out = os.path.join(HERE, "tlbr_reproduction.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
