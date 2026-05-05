#!/bin/bash
# Submit all sampler jobs for LinearRegression
# Usage: ./submit_all.sh [--resume N]
#   --resume N: resume runs with N additional iterations

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

RESUME=""
if [ "$1" = "--resume" ] && [ -n "$2" ]; then
    RESUME="$2"
    echo "Resuming jobs for LinearRegression with $RESUME additional iterations..."
else
    echo "Submitting fresh jobs for LinearRegression..."
fi

for sampler in am dream p_dream; do
    echo "  Submitting $sampler..."
    if [ -n "$RESUME" ]; then
        RESUME_ITERS="$RESUME" sbatch run_${sampler}.sh
    else
        sbatch run_${sampler}.sh
    fi
done
echo "All jobs submitted. Use 'squeue -u $USER' to monitor."
