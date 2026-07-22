#!/bin/sh
# Generate the three mutant/inhibitor model copies from the WT baseline.
# A BPSL constraint experiment can't apply a non-free-parameter condition, so (following
# the shipped Miller2025 one-model-per-condition pattern) each mutant/inhibitor is baked
# into its own copy of the model. Deltas are the published values (Supplementary Table 2).
base=phosphoswitch_bpsl.bngl

# JNK inhibitor (JNK-IN-8): JNK kinase activity abolished -> k1 = 0, k2 = 0.
perl -0pe 's/^(\s*k1\s+)[\d.eE+-]+/${1}0/m; s/^(\s*k2\s+)[\d.eE+-]+/${1}0/m' \
    "$base" > phosphoswitch_bpsl_jnki.bngl

# S90N: F-site serine->asparagine -> no S90 phosphorylation (k2 = 0) and tighter bipartite
# p38 binding (kon2 17.2 -> 77.1, from the Fig. 4b Ki, Table 2 note a).
perl -0pe 's/^(\s*kon2\s+)[\d.eE+-]+/${1}77.1/m; s/^(\s*k2\s+)[\d.eE+-]+/${1}0/m' \
    "$base" > phosphoswitch_bpsl_s90n.bngl

# MUT4: F-motif (SPFENEF) mutant -> crippled p38 FRS binding (kon2 17.2 -> 1.2, Table 2 note b).
perl -0pe 's/^(\s*kon2\s+)[\d.eE+-]+/${1}1.2/m' \
    "$base" > phosphoswitch_bpsl_mut4.bngl

echo "generated jnki (k1=k2=0), s90n (kon2=77.1,k2=0), mut4 (kon2=1.2) from $base"
