#!/bin/bash
# Submit all sampler benchmarking jobs to SLURM
# Usage: ./run_all.sh [--problems PROB1 PROB2 ...] [--samplers am dream p_dream] [--resume N]
#
# Examples:
#   ./run_all.sh                              # submit everything
#   ./run_all.sh --problems Banana HIVdynamics # submit only these problems
#   ./run_all.sh --samplers am p_dream         # submit only these samplers
#   ./run_all.sh --resume 5000                 # resume all with 5000 additional iterations

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ALL_PROBLEMS="Banana Gaussian_d10 Multimodal LinearRegression HIVdynamics COVID19_BigApple Degranulation TCR FcERI_gamma MEK_Isoforms EGFR_d10 EGFR_d37"
ALL_SAMPLERS="am dream p_dream"
PROBLEMS=""
SAMPLERS=""
RESUME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --problems) shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do PROBLEMS="$PROBLEMS $1"; shift; done ;;
        --samplers) shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SAMPLERS="$SAMPLERS $1"; shift; done ;;
        --resume) RESUME="$2"; shift; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PROBLEMS="${PROBLEMS:-$ALL_PROBLEMS}"
SAMPLERS="${SAMPLERS:-$ALL_SAMPLERS}"

echo "============================================"
echo "  Sampler Benchmarking - Job Submission"
echo "============================================"
echo "Problems: $PROBLEMS"
echo "Samplers: $SAMPLERS"
if [ -n "$RESUME" ]; then
    echo "Mode:     RESUME (adding $RESUME iterations)"
else
    echo "Mode:     FRESH"
fi
echo ""

submitted=0
for prob in $PROBLEMS; do
    if [ ! -d "$SCRIPT_DIR/$prob" ]; then
        echo "WARNING: $prob directory not found, skipping."
        continue
    fi
    cd "$SCRIPT_DIR/$prob"
    for samp in $SAMPLERS; do
        script="run_${samp}.sh"
        if [ ! -f "$script" ]; then
            echo "  WARNING: $prob/$script not found, skipping."
            continue
        fi
        echo "  Submitting $prob / $samp ..."
        if [ -n "$RESUME" ]; then
            RESUME_ITERS="$RESUME" sbatch "$script"
        else
            sbatch "$script"
        fi
        submitted=$((submitted + 1))
    done
done

echo ""
echo "$submitted jobs submitted. Monitor with: squeue -u $USER"
