#!/usr/bin/env python3
"""
Run sampler comparison experiments.

Experiments:
  1. Gaussian d=10:   Correctness + efficiency on a moderate-dimensional target
  2. Banana:          Correlated (banana-shaped) target
  3. Multimodal:      Mixture of Gaussians
  4. EGFR (optional): Real-world model (d=37), requires BioNetGen

The script avoids re-running a benchmark that already has results unless
--force is given.

Usage:
    python run_all_experiments.py [--replicates N] [--parallel P] [--force]
    python run_all_experiments.py --experiments 1 2 3    # run subset
    python run_all_experiments.py --skip-egfr             # skip slow real-world test
"""

import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_BENCHMARK = os.path.join(SCRIPT_DIR, 'run_benchmark.py')

# Experiment definitions: (name, benchmark_dir, description, extra_args)
ANALYTICAL_EXPERIMENTS = [
    ('gaussian_d10', 'gaussian_d10', 'Gaussian d=10 (correctness + efficiency)', {}),
    ('banana',       'banana',       'Correlated (banana) target', {}),
    ('multimodal',   'multimodal',   'Multimodal (mixture) target', {}),
]

EGFR_EXPERIMENT = ('egfr', 'egfr', 'Real-world EGFR (d=37)', {})

# Map experiment numbers from the roadmap to benchmark runs
EXPERIMENT_MAP = {
    1: ['gaussian_d10'],                                         # Correctness + efficiency
    2: ['banana'],                                               # Correlated
    3: ['multimodal'],                                           # Multimodal
    4: ['egfr'],                                                 # Real-world
}


def has_results(bench_dir):
    """Check if a benchmark already has results."""
    results_path = os.path.join(bench_dir, 'runs', 'results.json')
    return os.path.isfile(results_path)


def run_benchmark(bench_dir, replicates, parallel=None, force=False):
    """Run a single benchmark via run_benchmark.py."""
    bench_path = os.path.join(SCRIPT_DIR, bench_dir)
    if not os.path.isdir(bench_path):
        print('  ERROR: directory %s not found, skipping' % bench_path)
        return False

    if has_results(bench_path) and not force:
        print('  Already has results (use --force to re-run)')
        return True

    cmd = [sys.executable, RUN_BENCHMARK, bench_path,
           '--replicates', str(replicates)]
    if parallel is not None:
        cmd += ['--parallel', str(parallel)]

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


def load_results(bench_dir):
    """Load results.json for a benchmark."""
    path = os.path.join(SCRIPT_DIR, bench_dir, 'runs', 'results.json')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _print_experiment(title, results):
    """Print per-sampler summary for one experiment."""
    import numpy as np
    print('\n--- %s ---' % title)
    if not results:
        print('  (not yet run)')
        return
    # Whichever samplers actually produced results, in first-seen order.
    for sampler in dict.fromkeys(r['sampler'] for r in results):
        runs = [r for r in results if r['sampler'] == sampler and r.get('success')]
        if not runs:
            continue
        n = len(runs)
        wall = np.mean([r['wall_clock'] for r in runs])
        ess_eval = [r['min_ess_per_eval'] for r in runs
                    if 'min_ess_per_eval' in r and np.isfinite(r['min_ess_per_eval'])]
        rhat = [r['max_rhat'] for r in runs if 'max_rhat' in r]
        d_vals = [r['distance_D'] for r in runs if 'distance_D' in r]
        parts = ['  %s (N=%d): wall=%.0fs' % (sampler, n, wall)]
        if rhat:
            parts.append('R-hat=%.3f+-%.3f' % (np.mean(rhat), np.std(rhat)))
        if ess_eval:
            parts.append('ESS/eval=%.4f+-%.4f' % (np.mean(ess_eval), np.std(ess_eval)))
        if d_vals:
            parts.append('D=%.3f+-%.3f' % (np.mean(d_vals), np.std(d_vals)))
        print(', '.join(parts))


def print_summary():
    """Print a combined summary of all experiments."""
    print('\n')
    print('#' * 100)
    print('#  EXPERIMENT SUMMARY')
    print('#' * 100)

    _print_experiment('Experiment 1: Gaussian d=10', load_results('gaussian_d10'))
    _print_experiment('Experiment 2: Correlated target (Banana)', load_results('banana'))
    _print_experiment('Experiment 3: Multimodal (Mixture of Gaussians)', load_results('multimodal'))
    _print_experiment('Experiment 4: Real-world EGFR (d=37)', load_results('egfr'))

    print('\n' + '#' * 100)


def main():
    parser = argparse.ArgumentParser(
        description='Run all Phase 4 sampler comparison experiments')
    parser.add_argument('--replicates', '-n', type=int, default=5,
                        help='Replicates per sampler (default: 5)')
    parser.add_argument('--parallel', '-p', type=int, default=None,
                        help='Parallel workers for PyBNF')
    parser.add_argument('--force', action='store_true',
                        help='Re-run benchmarks even if results exist')
    parser.add_argument('--experiments', '-e', type=int, nargs='+',
                        default=None,
                        help='Run only specific experiments (1-6)')
    parser.add_argument('--skip-egfr', action='store_true',
                        help='Skip the EGFR real-world experiment')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only print summary from existing results')
    args = parser.parse_args()

    if args.summary_only:
        print_summary()
        return

    # Determine which benchmarks to run
    if args.experiments:
        bench_dirs = set()
        for exp_num in args.experiments:
            if exp_num not in EXPERIMENT_MAP:
                print('Unknown experiment %d (valid: 1-4)' % exp_num)
                sys.exit(1)
            bench_dirs.update(EXPERIMENT_MAP[exp_num])
    else:
        bench_dirs = {name for name, _, _, _ in ANALYTICAL_EXPERIMENTS}
        if not args.skip_egfr:
            bench_dirs.add('egfr')

    # Run experiments
    all_experiments = ANALYTICAL_EXPERIMENTS + [EGFR_EXPERIMENT]
    t_total = time.time()

    for name, bench_dir, desc, extra_args in all_experiments:
        if name not in bench_dirs:
            continue

        print('\n' + '=' * 70)
        print('EXPERIMENT: %s (%s)' % (name, desc))
        print('=' * 70)

        t0 = time.time()
        success = run_benchmark(bench_dir, args.replicates, args.parallel,
                                force=args.force)
        elapsed = time.time() - t0

        status = 'OK' if success else 'FAILED'
        print('[%s] %s completed in %.1fs' % (status, name, elapsed))

    total_time = time.time() - t_total
    print('\n\nAll experiments completed in %.1fs' % total_time)

    # Print combined summary
    print_summary()


if __name__ == '__main__':
    main()
