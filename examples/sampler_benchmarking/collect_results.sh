#!/bin/bash
# Collect and display results from sampler benchmarking runs
# Usage: ./collect_results.sh [--problems PROB1 PROB2 ...]
#
# Checks each problem/sampler combination for:
#   - Completion status (alg_finished.bp vs alg_backup.bp)
#   - Convergence diagnostics (R-hat, Bulk ESS from diagnostics.txt)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ALL_PROBLEMS="Banana Gaussian_d10 Multimodal LinearRegression HIVdynamics COVID19_BigApple Degranulation TCR FcERI_gamma MEK_Isoforms EGFR_d10 EGFR_d37"
ALL_SAMPLERS="am dream p_dream"
PROBLEMS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --problems) shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do PROBLEMS="$PROBLEMS $1"; shift; done ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PROBLEMS="${PROBLEMS:-$ALL_PROBLEMS}"

# Pretty sampler names for display
sampler_label() {
    case "$1" in
        am)       echo "AM        " ;;
        dream)    echo "DREAM(ZS) " ;;
        p_dream)  echo "P-DREAM   " ;;
        *)        echo "$1         " ;;
    esac
}

echo "============================================"
echo "  Sampler Benchmarking - Results Summary"
echo "============================================"

for prob in $PROBLEMS; do
    prob_dir="$SCRIPT_DIR/$prob"
    if [ ! -d "$prob_dir" ]; then
        continue
    fi

    echo ""
    echo "=== $prob ==="
    printf "  %-12s %-22s %-14s %-14s %s\n" "Sampler" "Status" "Max R-hat" "Min Bulk ESS" "Evals"
    printf "  %-12s %-22s %-14s %-14s %s\n" "-------" "------" "---------" "------------" "-----"

    for samp in $ALL_SAMPLERS; do
        outdir="$prob_dir/output_${samp}"
        label=$(sampler_label "$samp")

        if [ ! -d "$outdir" ]; then
            printf "  %-12s %-22s\n" "$label" "NOT YET RUN"
            continue
        fi

        # Check completion status
        if [ -f "$outdir/alg_finished.bp" ]; then
            status="COMPLETED"
        elif [ -f "$outdir/alg_backup.bp" ]; then
            status="IN PROGRESS"
        else
            status="UNKNOWN"
        fi

        # Parse diagnostics if available
        diag="$outdir/Results/diagnostics.txt"
        if [ -f "$diag" ]; then
            # Read the header to find column indices
            header=$(head -1 "$diag" | sed 's/^# //')

            # Get the last data line
            last_line=$(tail -1 "$diag")

            # Extract total_evaluations (column 2)
            evals=$(echo "$last_line" | awk '{print $2}')

            # Find max R-hat and min Bulk ESS across all parameters
            # R-hat columns contain "rhat_", Bulk ESS columns contain "bulk_ess_"
            # Use awk to parse header and find relevant columns
            max_rhat=$(echo "$last_line" | awk -v header="$header" '
                BEGIN { split(header, h, "\t"); max=-1 }
                {
                    split($0, v, "\t")
                    for (i in h) {
                        if (h[i] ~ /^rhat_/) {
                            val = v[i] + 0
                            if (val > max || max == -1) max = val
                        }
                    }
                    printf "%.4f", max
                }
            ')

            min_ess=$(echo "$last_line" | awk -v header="$header" '
                BEGIN { split(header, h, "\t"); min=-1 }
                {
                    split($0, v, "\t")
                    for (i in h) {
                        if (h[i] ~ /^bulk_ess_/) {
                            val = v[i] + 0
                            if (val < min || min == -1) min = val
                        }
                    }
                    printf "%.1f", min
                }
            ')

            printf "  %-12s %-22s %-14s %-14s %s\n" "$label" "$status" "$max_rhat" "$min_ess" "$evals"
        else
            printf "  %-12s %-22s %s\n" "$label" "$status" "no diagnostics yet"
        fi
    done
done

echo ""
echo "Done."
