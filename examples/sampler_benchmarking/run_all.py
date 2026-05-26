#!/usr/bin/env python3
"""
Submit all sampler benchmarking jobs to SLURM (or run locally).

Usage:
    python run_all.py                                  # submit everything via sbatch
    python run_all.py --problems Banana HIVdynamics    # submit only these problems
    python run_all.py --samplers am p_dream            # submit only these samplers
    python run_all.py --dry-run                        # print what would be submitted
    python run_all.py --local                          # run pybnf directly (no SLURM)
    python run_all.py --local --problems Banana        # run locally for one problem
    python run_all.py --resume 5000                    # resume all with 5000 additional iterations
    python run_all.py --resume 0                       # resume to original max_iterations
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_PROBLEMS = [
    "Banana", "Gaussian_d10", "Multimodal", "LinearRegression",
    "HIVdynamics", "COVID19_BigApple", "Degranulation",
    "TCR", "FcERI_gamma", "MEK_Isoforms",
    "EGFR_d10", "EGFR_d37",
]
ALL_SAMPLERS = ["am", "dream", "p_dream"]


def build_local_command(conf_file, uses_bng=False):
    """Build the pybnf command for local (non-SLURM) execution."""
    # Analytical targets don't need -t SLURM; BNG models do for parallelism
    if uses_bng:
        return f"pybnf -c {conf_file} -t SLURM -o"
    else:
        return f"pybnf -c {conf_file} -o"


# Problems that require BioNetGen (and thus benefit from -t SLURM)
BNG_PROBLEMS = {"LinearRegression", "HIVdynamics", "COVID19_BigApple",
                "Degranulation", "TCR", "FcERI_gamma", "MEK_Isoforms",
                "EGFR_d10", "EGFR_d37"}


def main():
    parser = argparse.ArgumentParser(
        description="Submit sampler benchmarking jobs to SLURM or run locally."
    )
    parser.add_argument(
        "--problems", nargs="+", default=ALL_PROBLEMS, metavar="PROB",
        help="Problem names to run (default: all)",
    )
    parser.add_argument(
        "--samplers", nargs="+", default=ALL_SAMPLERS, metavar="SAMP",
        help="Sampler names to run (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be submitted without actually submitting.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run pybnf directly instead of submitting via sbatch (sequential).",
    )
    parser.add_argument(
        "--resume", type=int, default=None, metavar="N",
        help="Resume runs with N additional iterations (0 = resume to original max_iterations).",
    )
    args = parser.parse_args()

    print("=" * 48)
    print("  Sampler Benchmarking - Job Submission")
    print("=" * 48)
    print(f"  Problems: {' '.join(args.problems)}")
    print(f"  Samplers: {' '.join(args.samplers)}")
    if args.dry_run:
        print("  Mode:     DRY RUN (no jobs will be submitted)")
    elif args.local:
        print("  Mode:     LOCAL (running pybnf directly)")
    else:
        print("  Mode:     SLURM (sbatch)")
    if args.resume is not None:
        print(f"  Resume:   {args.resume} additional iterations")
    print()

    submitted = 0
    failed = 0

    for prob in args.problems:
        prob_dir = os.path.join(SCRIPT_DIR, prob)
        if not os.path.isdir(prob_dir):
            print(f"  WARNING: {prob} directory not found, skipping.")
            continue

        for samp in args.samplers:
            script = os.path.join(prob_dir, f"run_{samp}.sh")
            if not os.path.isfile(script):
                print(f"  WARNING: {prob}/run_{samp}.sh not found, skipping.")
                continue

            uses_bng = prob in BNG_PROBLEMS

            if args.dry_run:
                if args.local:
                    if args.resume is not None:
                        cmd = f"pybnf -r {args.resume} -c {samp}.conf"
                    else:
                        cmd = build_local_command(f"{samp}.conf", uses_bng)
                    print(f"  [DRY RUN] Would run in {prob}/: {cmd}")
                else:
                    if args.resume is not None:
                        print(f"  [DRY RUN] Would submit: RESUME_ITERS={args.resume} sbatch {script}")
                    else:
                        print(f"  [DRY RUN] Would submit: sbatch {script}")
                submitted += 1
                continue

            if args.local:
                if args.resume is not None:
                    cmd = f"pybnf -r {args.resume} -c {samp}.conf"
                else:
                    cmd = build_local_command(f"{samp}.conf", uses_bng)
                print(f"  Running {prob} / {samp}: {cmd}")
                try:
                    subprocess.run(
                        cmd, shell=True, cwd=prob_dir, check=True,
                    )
                    submitted += 1
                    print(f"  Completed {prob} / {samp}")
                except subprocess.CalledProcessError as e:
                    print(f"  FAILED {prob} / {samp}: exit code {e.returncode}")
                    failed += 1
            else:
                # Submit via sbatch
                print(f"  Submitting {prob} / {samp} ...")
                try:
                    env = None
                    if args.resume is not None:
                        env = {**os.environ, "RESUME_ITERS": str(args.resume)}
                    result = subprocess.run(
                        ["sbatch", script], cwd=prob_dir, env=env,
                        capture_output=True, text=True, check=True,
                    )
                    print(f"    {result.stdout.strip()}")
                    submitted += 1
                except FileNotFoundError:
                    print("  ERROR: sbatch not found. Use --local to run without SLURM.")
                    sys.exit(1)
                except subprocess.CalledProcessError as e:
                    print(f"  FAILED to submit {prob} / {samp}: {e.stderr.strip()}")
                    failed += 1

    print()
    if args.dry_run:
        print(f"{submitted} jobs would be submitted.")
    elif args.local:
        print(f"{submitted} jobs completed, {failed} failed.")
    else:
        print(f"{submitted} jobs submitted, {failed} failed.")
        print("Monitor with: squeue -u $USER")


if __name__ == "__main__":
    main()
