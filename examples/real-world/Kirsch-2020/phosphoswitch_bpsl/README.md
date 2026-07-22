# phosphoswitch_bpsl — S90 phosphoswitch / p38 recruitment BPSL job (PyBNF, native-only)

This example exercises PyBNF's **signature capability — BPSL** (the
Biological Property Specification Language). Instead of fitting numbers, it fits the
paper's **central qualitative claim** about how the ATF2 phosphoswitch controls p38 binding.

> Kirsch K, Zeke A, Tőke O, Sok P, Sethi A, Sebő A, Kumar GS, Egri P, Póti ÁL,
> Gooley P, Peti W, Bento I, Reményi A, Alexa A.
> **"Co-regulation of the transcription controlling ATF2 phosphoswitch by JNK and p38."**
> *Nat Commun* 2020; **11**(1):5769.
> PMCID: [PMC7666158](https://pmc.ncbi.nlm.nih.gov/articles/PMC7666158/) ·
> DOI: [10.1038/s41467-020-19582-3](https://doi.org/10.1038/s41467-020-19582-3)

## The idea

The paper's headline is a **phosphoswitch**: JNK phosphorylates the ATF2 F-site **S90**,
and **S90-P sterically blocks p38's F-site (FRS) recruitment**. So JNK activity feeds back
to *diminish* p38 binding — one MAPK shapes how the other engages ATF2. The model
reproduces this at the published parameters through the total p38:ATF2 TAD complex
(`p38ATF2all`, the NanoBit binding readout) across four conditions:

| condition (Suppl. Table 2 deltas) | steady-state `p38ATF2all` (µM) | p38 recruitment |
|---|---:|---|
| **S90N** (kon2 17.2→77.1, k2=0) | **0.0250** | highest — F-site block removed |
| **JNK inhibitor** (k1=k2=0) | **0.0073** | raised — S90-P block relieved |
| WT | 0.0031 | reference |
| **MUT4** (kon2→1.2) | **0.0022** | lowest — F-motif crippled |

`S90N > JNKi > WT > MUT4`. Those qualitative facts are written as **BPSL `.prop`
constraints on `p38ATF2all`** (see `phosphoswitch_bpsl_switch.png`). The paper reports the
binding quantitatively (Ki values, Fig. 4b; enzyme kinetics, Fig. 3c; NanoBit, Figs. 3–4);
expressing the mechanism as cross-condition BPSL orderings is this example's contribution.

## The constraints (`.prop` files)

```
wt.prop:    p38ATF2all > 0.001 at 20000 weight 1        # p38 does recruit to WT ATF2
            pT69pT71   > 0.5   at 20000 weight 1        # WT reaches substantial phosphorylation
jnki.prop:  p38ATF2all > wt.p38ATF2all at 20000 weight 10    # JNKi relieves the S90-P block
s90n.prop:  p38ATF2all > wt.p38ATF2all at 20000 weight 10    # S90N: F-site block removed
            p38ATF2all > jnki.p38ATF2all at 20000 weight 5   #   ...and above pharmacological JNKi
mut4.prop:  p38ATF2all < wt.p38ATF2all at 20000 weight 8     # F-motif mutant -> less binding
```

`wt.p38ATF2all` / `jnki.p38ATF2all` are **cross-experiment (dotted-suffix)** references —
the suffix is the experiment name. `at 20000` reads the settled stimulated steady state.

## Structure — one model per condition (Miller2025 pattern)

A BPSL constraint-only experiment cannot apply a **non-free-parameter condition**
(`preequilibrate:` needs an `.exp`, and the no-preequilibration path mutates only *free*
params). As in the shipped Miller MEK-isoform example, each mutant/inhibitor is
**baked into its own model copy** (with `Stim = 1`, the stimulated steady state), and
each experiment binds that model:

| file | delta (Suppl. Table 2) | experiment | constraints |
|---|---|---|---|
| `phosphoswitch_bpsl.bngl` | none (WT) | `wt` | `wt.prop` |
| `phosphoswitch_bpsl_jnki.bngl` | `k1=0, k2=0` | `jnki` | `jnki.prop` |
| `phosphoswitch_bpsl_s90n.bngl` | `kon2=77.1, k2=0` | `s90n` | `s90n.prop` |
| `phosphoswitch_bpsl_mut4.bngl` | `kon2=1.2` | `mut4` | `mut4.prop` |

The three variant models are generated from the WT baseline by **`make_models.sh`**. The
four free cell params bind by name across all models.

## Free parameters (4)

Cell-level rate constants that shift where p38 recruitment settles, `loguniform` over ≈±½
decade around the published value: `dp2` (1.76e-3, pp-p38 phosphatase), `dp4` (4.50e-3,
pS90 phosphatase — how fast the S90-P block is reversed), `kstim6` (1.16e-4, p38
activation), `kstim7` (9.074e-5, JNK activation — drives S90 phosphorylation). The mutant
deltas (`kon2`, `k1`, `k2`) that *produce* the orderings are baked into the models, not fit.

## ⚠️ Native-only (not PEtab-v2-exportable)

Attaching **any** `.prop`/`.con` makes the job native-only: `export_job` raises
`NotImplementedError` (verified — BPSL has no PEtab v2 representation). Verify with
`job_type = check`, **not** the PEtab round-trip.

## Verification (all pass)

- **Tier-1** (`check_conf.py`): edition 2, `job_type=de` resolves, **4 BPSL constraint sets
  bound**, 4 free params bind by id, no `__FREE`; correctly reported native-only. **PASS.**
- **Model build** (BNG2.pl): each model generates **48 species / 152 reactions**.
- **`job_type = check` at published params** (`phosphoswitch_bpsl_check.conf`):
  **`Satisfied 6 out of 6 constraints`**, objective **0.0** — the model satisfies every
  qualitative claim, including the cross-condition comparisons.
- **Bounded bngsim `de` fit** (constraints as soft penalties): finite objective (0.0) — the
  simulate→score→propose loop runs across all four models.
- **Native-only assertion**: `pybnf.petab.export_job(...)` raises `NotImplementedError`.
- **Phosphoswitch** (`phosphoswitch_bpsl_switch.png`): p38:ATF2 recruitment reaches
  0.0250 / 0.0073 / 0.0031 / 0.0022 µM under S90N / JNKi / WT / MUT4 — S90N and JNK
  inhibition raise p38 binding above WT (the block relieved), MUT4 lowers it.

## Run

```bash
export BNGPATH="$HOME/Simulations/BioNetGen-2.9.3"   # folder with BNG2.pl
cd examples/real-world/Kirsch-2020/phosphoswitch_bpsl
./make_models.sh                                     # (re)generate the mutant models
pybnf -c phosphoswitch_bpsl.conf                     # fit the cell params to the qualitative facts
pybnf -c phosphoswitch_bpsl_check.conf               # or: check satisfaction at published params
```

## `_manifest.py` note (if promoted to the PyBNF real-world corpus)

```python
RealWorldExample(
    folder='Kirsch-2020/phosphoswitch_bpsl', conf='phosphoswitch_bpsl.conf', simulator='ode',
    observables=('p38ATF2all',),
    system='JNK/p38/ATF2 S90 phosphoswitch controlling p38 recruitment (Kirsch 2020, '
           'PMC7666158); ODE; BPSL .prop constraints on p38ATF2all -> NATIVE-ONLY (not '
           'PEtab-exportable); S90N/JNKi raise p38 binding above WT, MUT4 lowers it'),
    # BPSL: assert export_job raises NotImplementedError, and/or a `check` run reports
    # "Satisfied 6 out of 6" -- do NOT add a PEtab-lint test (it will correctly refuse).
```
