#!/usr/bin/env python
"""Phosphoswitch figure for the phosphoswitch_bpsl job (PMC7666158).

Runs the four condition models (WT, JNK inhibitor, S90N, MUT4) to the stimulated steady
state and plots p38:ATF2 TAD recruitment (p38ATF2all) -- the readout the BPSL .prop
constraints act on. Shows the paper's central qualitative claim: JNK phosphorylation of
S90 blocks p38's F-site recruitment, so removing it (S90N) or inhibiting JNK raises p38
binding above WT, while crippling the F-motif (MUT4) lowers it. Requires BNGPATH + matplotlib.

Usage:  BNGPATH=... ./make_models.sh && python make_switch.py
"""
import os
import subprocess
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BNG = os.path.join(os.environ["BNGPATH"], "BNG2.pl")
CONDS = [("phosphoswitch_bpsl.bngl", "WT", "#4c72b0"),
         ("phosphoswitch_bpsl_jnki.bngl", "JNK inhibitor", "#dd8452"),
         ("phosphoswitch_bpsl_s90n.bngl", "S90N", "#c44e52"),
         ("phosphoswitch_bpsl_mut4.bngl", "MUT4", "#8172b3")]
T_END = 20000


def run(model):
    with open(os.path.join(HERE, model)) as fh:
        src = fh.read().split("end model")[0] + "end model\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.bngl")
        with open(path, "w") as fh:
            fh.write(src)
            fh.write('\ngenerate_network({overwrite=>1})\n')
            fh.write(f'simulate({{prefix=>"s",method=>"ode",t_start=>0,t_end=>{T_END},n_steps=>400}})\n')
        subprocess.run(["perl", BNG, path], cwd=d, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return np.loadtxt(os.path.join(d, "s.gdat"))  # time pT69pT71 p38ATF2all ppp38 ppJNK


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ss = {}
    for model, lab, color in CONDS:
        g = run(model)
        ss[lab] = g[-1, 2]
        axL.plot(g[:, 0] / 60, g[:, 2], "-", color=color, lw=2, label=lab)
    axL.set_xlabel("time under sustained stimulation (min)")
    axL.set_ylabel("p38:ATF2 TAD complex, p38ATF2all (uM)")
    axL.set_title("p38 recruitment trajectories")
    axL.legend(frameon=False, fontsize=9)
    axL.grid(alpha=0.25)

    labs = [c[1] for c in CONDS]
    colors = [c[2] for c in CONDS]
    vals = [ss[l] for l in labs]
    axR.bar(labs, vals, color=colors)
    for i, v in enumerate(vals):
        axR.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    axR.axhline(ss["WT"], ls="--", color="#4c72b0", lw=1, alpha=0.7)
    axR.set_ylabel("steady-state p38ATF2all (uM)")
    axR.set_title("S90N > JNKi > WT > MUT4  (BPSL constraints)")
    axR.tick_params(axis="x", rotation=20)
    axR.grid(alpha=0.25, axis="y")

    fig.suptitle("Kirsch 2020 (PMC7666158) -- S90 phosphoswitch controls p38 recruitment to ATF2", y=1.02)
    fig.tight_layout()
    out = os.path.join(HERE, "phosphoswitch_bpsl_switch.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)
    print("steady-state p38ATF2all:", {k: round(v, 5) for k, v in ss.items()})
    print("orderings: s90n>jnki>wt>mut4 ->",
          ss["S90N"] > ss["JNK inhibitor"] > ss["WT"] > ss["MUT4"])


if __name__ == "__main__":
    main()
