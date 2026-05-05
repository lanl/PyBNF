#!/usr/bin/env python3
"""
Collect and display results from sampler benchmarking runs.

Usage:
    python collect_results.py                              # show all results
    python collect_results.py --problems Banana Multimodal # specific problems
    python collect_results.py --samplers am dream          # specific samplers
    python collect_results.py --output-csv results.csv     # save to CSV
    python collect_results.py --output-json results.json   # save to JSON
    python collect_results.py --compare                    # rank samplers per problem
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_PROBLEMS = [
    "Banana", "Gaussian_d10", "Multimodal", "LinearRegression",
    "HIVdynamics", "COVID19_BigApple", "Degranulation",
    "TCR", "FcERI_gamma", "MEK_Isoforms",
    "EGFR_d10", "EGFR_d37",
]
ALL_SAMPLERS = ["am", "dream", "p_dream"]

SAMPLER_LABELS = {
    "am": "AM",
    "dream": "DREAM(ZS)",
    "p_dream": "P-DREAM",
}

# ANSI color codes
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"


def colorize(text, color, use_color=True):
    if not use_color:
        return text
    return f"{color}{text}{COLOR_RESET}"


def pad_colored(text, width, use_color=True):
    """Pad text to width, accounting for invisible ANSI escape sequences."""
    if not use_color:
        return f"{text:<{width}}"
    # Strip ANSI codes to measure visible length
    visible = re.sub(r'\033\[[0-9;]*m', '', text)
    padding = max(0, width - len(visible))
    return text + ' ' * padding


def parse_diagnostics(output_dir):
    """Parse diagnostics.txt and return a dict with summary statistics.

    The file is tab-separated with a header line starting with '# '.
    Columns include: iteration, total_evaluations, and per-parameter
    rhat_<name>, bulk_ess_<name>, tail_ess_<name>.

    Returns the final row parsed into a dict with computed summaries:
      max_rhat, min_bulk_ess, min_tail_ess, total_evaluations, iteration
    """
    diag_file = os.path.join(output_dir, "Results", "diagnostics.txt")
    if not os.path.isfile(diag_file):
        return None

    with open(diag_file) as f:
        header_line = f.readline().lstrip("# ").strip()
        headers = header_line.split("\t")
        last_line = None
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if last_line is None:
        return None

    vals = last_line.split("\t")
    data = {}
    for h, v in zip(headers, vals):
        try:
            data[h] = float(v)
        except ValueError:
            data[h] = float("nan")

    # Compute summary statistics across parameters
    rhats = [data[h] for h in headers if h.startswith("rhat_") and not math.isnan(data.get(h, float("nan")))]
    bulk_ess = [data[h] for h in headers if h.startswith("bulk_ess_") and not math.isnan(data.get(h, float("nan")))]
    tail_ess = [data[h] for h in headers if h.startswith("tail_ess_") and not math.isnan(data.get(h, float("nan")))]

    result = {
        "iteration": int(data.get("iteration", 0)),
        "total_evaluations": int(data.get("total_evaluations", 0)),
        "max_rhat": max(rhats) if rhats else float("nan"),
        "min_bulk_ess": min(bulk_ess) if bulk_ess else float("nan"),
        "min_tail_ess": min(tail_ess) if tail_ess else float("nan"),
        "mean_rhat": sum(rhats) / len(rhats) if rhats else float("nan"),
        "mean_bulk_ess": sum(bulk_ess) / len(bulk_ess) if bulk_ess else float("nan"),
        # Per-parameter details
        "per_param": {
            h.replace("rhat_", ""): {
                "rhat": data.get(f"rhat_{h.replace('rhat_', '')}", float("nan")),
                "bulk_ess": data.get(f"bulk_ess_{h.replace('rhat_', '')}", float("nan")),
                "tail_ess": data.get(f"tail_ess_{h.replace('rhat_', '')}", float("nan")),
            }
            for h in headers if h.startswith("rhat_")
        },
    }
    return result


def parse_wall_time(prob_dir, sampler):
    """Try to extract wall time from SLURM output or log files."""
    # Look for SLURM output files matching the pattern
    patterns = [
        os.path.join(prob_dir, f"*{sampler}*.out"),
        os.path.join(prob_dir, f"output_{sampler}", "*.log"),
        os.path.join(prob_dir, f"output_{sampler}", "pybnf.log"),
    ]
    for pattern in patterns:
        for logfile in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True):
            try:
                with open(logfile) as f:
                    content = f.read()
                # Look for wall time patterns
                # PyBNF logs: "Total wall time: X seconds"
                m = re.search(r"Total wall time:\s*([\d.]+)\s*seconds", content)
                if m:
                    return float(m.group(1))
                # Also try "Elapsed time: HH:MM:SS"
                m = re.search(r"Elapsed.*?(\d+):(\d+):(\d+)", content)
                if m:
                    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            except (IOError, OSError):
                continue
    return None


def format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds is None:
        return "---"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def get_run_status(output_dir):
    """Determine run status from checkpoint files."""
    if not os.path.isdir(output_dir):
        return "NOT RUN"
    if os.path.isfile(os.path.join(output_dir, "alg_finished.bp")):
        return "COMPLETED"
    if os.path.isfile(os.path.join(output_dir, "alg_backup.bp")):
        return "IN PROGRESS"
    # Check if Results directory exists at least
    if os.path.isdir(os.path.join(output_dir, "Results")):
        return "PARTIAL"
    return "UNKNOWN"


def collect_all_results(problems, samplers):
    """Collect results for all problem/sampler combinations.

    Returns a list of dicts, one per combination.
    """
    results = []
    for prob in problems:
        prob_dir = os.path.join(SCRIPT_DIR, prob)
        if not os.path.isdir(prob_dir):
            continue

        for samp in samplers:
            output_dir = os.path.join(prob_dir, f"output_{samp}")
            status = get_run_status(output_dir)
            diag = parse_diagnostics(output_dir) if status != "NOT RUN" else None
            wall_time = parse_wall_time(prob_dir, samp)

            entry = {
                "problem": prob,
                "sampler": samp,
                "sampler_label": SAMPLER_LABELS.get(samp, samp),
                "status": status,
                "wall_time_s": wall_time,
                "wall_time_fmt": format_time(wall_time),
            }
            if diag:
                entry.update({
                    "max_rhat": diag["max_rhat"],
                    "min_bulk_ess": diag["min_bulk_ess"],
                    "min_tail_ess": diag["min_tail_ess"],
                    "mean_rhat": diag["mean_rhat"],
                    "mean_bulk_ess": diag["mean_bulk_ess"],
                    "total_evaluations": diag["total_evaluations"],
                    "iteration": diag["iteration"],
                })
            else:
                entry.update({
                    "max_rhat": None,
                    "min_bulk_ess": None,
                    "min_tail_ess": None,
                    "mean_rhat": None,
                    "mean_bulk_ess": None,
                    "total_evaluations": None,
                    "iteration": None,
                })
            results.append(entry)

    return results


def print_results_table(results, use_color=True):
    """Print a formatted results table to stdout."""
    # Group by problem
    problems_seen = []
    by_problem = {}
    for r in results:
        p = r["problem"]
        if p not in by_problem:
            by_problem[p] = []
            problems_seen.append(p)
        by_problem[p].append(r)

    for prob in problems_seen:
        print()
        print(colorize(f"=== {prob} ===", COLOR_BOLD, use_color))
        print(f"  {'Sampler':<12} {'Status':<18} {'Max R-hat':<12} {'Min Bulk ESS':<14} {'Evals':<10} {'Time':<8}")
        print(f"  {'-'*12} {'-'*18} {'-'*12} {'-'*14} {'-'*10} {'-'*8}")

        for r in by_problem[prob]:
            # Color-code status
            status = r["status"]
            if status == "COMPLETED":
                status_str = colorize(status, COLOR_GREEN, use_color)
            elif status in ("IN PROGRESS", "PARTIAL"):
                status_str = colorize(status, COLOR_YELLOW, use_color)
            elif status == "NOT RUN":
                status_str = status
            else:
                status_str = colorize(status, COLOR_RED, use_color)

            # Format metrics
            if r["max_rhat"] is not None and not math.isnan(r["max_rhat"]):
                rhat_str = f"{r['max_rhat']:.4f}"
                # Color-code R-hat: green if < 1.1, yellow if < 1.2, red otherwise
                rhat_val = r["max_rhat"]
                if rhat_val < 1.05:
                    rhat_str = colorize(rhat_str, COLOR_GREEN, use_color)
                elif rhat_val < 1.1:
                    rhat_str = colorize(rhat_str, COLOR_YELLOW, use_color)
                else:
                    rhat_str = colorize(rhat_str, COLOR_RED, use_color)
            else:
                rhat_str = "---"

            if r["min_bulk_ess"] is not None and not math.isnan(r["min_bulk_ess"]):
                ess_str = f"{r['min_bulk_ess']:.0f}"
            else:
                ess_str = "---"

            evals_str = str(r["total_evaluations"]) if r["total_evaluations"] is not None else "---"
            time_str = r["wall_time_fmt"]

            label = r["sampler_label"]
            # Use ANSI-aware padding for colored fields
            print(f"  {label:<12} "
                  f"{pad_colored(status_str, 18, use_color)} "
                  f"{pad_colored(rhat_str, 12, use_color)} "
                  f"{ess_str:<14} {evals_str:<10} {time_str:<8}")


def print_comparison(results, use_color=True):
    """Print a comparison ranking samplers per problem."""
    print()
    print(colorize("=== Sampler Comparison (ranked by min Bulk ESS, higher is better) ===", COLOR_BOLD, use_color))

    by_problem = {}
    problems_seen = []
    for r in results:
        p = r["problem"]
        if p not in by_problem:
            by_problem[p] = []
            problems_seen.append(p)
        by_problem[p].append(r)

    for prob in problems_seen:
        entries = by_problem[prob]
        # Filter to completed runs with diagnostics
        ranked = [
            r for r in entries
            if r["status"] == "COMPLETED" and r["min_bulk_ess"] is not None
            and not math.isnan(r["min_bulk_ess"])
        ]
        if not ranked:
            print(f"\n  {prob}: no completed runs with diagnostics")
            continue

        ranked.sort(key=lambda r: r["min_bulk_ess"], reverse=True)
        print(f"\n  {prob}:")
        for i, r in enumerate(ranked, 1):
            converged = r["max_rhat"] < 1.1 if r["max_rhat"] is not None and not math.isnan(r["max_rhat"]) else False
            conv_tag = "" if converged else colorize(" [NOT CONVERGED]", COLOR_RED, use_color)
            evals = r["total_evaluations"] if r["total_evaluations"] is not None else "?"
            print(
                f"    {i}. {r['sampler_label']:<12} "
                f"Bulk ESS={r['min_bulk_ess']:>8.0f}  "
                f"R-hat={r['max_rhat']:>7.4f}  "
                f"Evals={evals}{conv_tag}"
            )


def save_csv(results, filepath):
    """Save results to a CSV file."""
    fieldnames = [
        "problem", "sampler", "sampler_label", "status",
        "max_rhat", "mean_rhat", "min_bulk_ess", "mean_bulk_ess", "min_tail_ess",
        "total_evaluations", "iteration", "wall_time_s",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            # Replace None with empty string for CSV
            row = {k: ("" if v is None else v) for k, v in r.items() if k in fieldnames}
            writer.writerow(row)
    print(f"\nResults saved to {filepath}")


def save_json(results, filepath):
    """Save results to a JSON file."""
    # Convert NaN to null for JSON compatibility
    clean = []
    for r in results:
        entry = {}
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                entry[k] = None
            else:
                entry[k] = v
        clean.append(entry)

    with open(filepath, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect and display sampler benchmarking results."
    )
    parser.add_argument(
        "--problems", nargs="+", default=ALL_PROBLEMS, metavar="PROB",
        help="Problem names to collect (default: all)",
    )
    parser.add_argument(
        "--samplers", nargs="+", default=ALL_SAMPLERS, metavar="SAMP",
        help="Sampler names to collect (default: all)",
    )
    parser.add_argument(
        "--output-csv", metavar="FILE",
        help="Save results to a CSV file.",
    )
    parser.add_argument(
        "--output-json", metavar="FILE",
        help="Save results to a JSON file.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Print a comparison ranking samplers per problem.",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output.",
    )
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    print(colorize("=" * 48, COLOR_BOLD, use_color))
    print(colorize("  Sampler Benchmarking - Results", COLOR_BOLD, use_color))
    print(colorize("=" * 48, COLOR_BOLD, use_color))

    results = collect_all_results(args.problems, args.samplers)

    if not results:
        print("\nNo results found.")
        return

    print_results_table(results, use_color)

    if args.compare:
        print_comparison(results, use_color)

    if args.output_csv:
        save_csv(results, args.output_csv)

    if args.output_json:
        save_json(results, args.output_json)


if __name__ == "__main__":
    main()
