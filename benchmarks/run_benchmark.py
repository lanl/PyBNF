#!/usr/bin/env python3
"""
Benchmark harness for MCMC sampler comparison.

Runs each sampler (am, dream, p_dream) N times on a given analytical target,
collects convergence diagnostics, and produces comparison tables and plots.

Usage:
    python run_benchmark.py <benchmark_dir> [--replicates N] [--parallel P]

Example:
    python run_benchmark.py gaussian_d5 --replicates 5
    python run_benchmark.py banana --replicates 10 --parallel 4
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

SAMPLERS = ['am', 'dream', 'p_dream']


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_single(conf_path, cwd=None, parallel=None, timeout=14400):
    """Run one PyBNF fit, returning wall-clock seconds and success flag."""
    env = os.environ.copy()
    cmd = [sys.executable, '-m', 'pybnf', '-c', conf_path, '-o']
    if parallel is not None:
        env['PYBNF_PARALLEL'] = str(parallel)

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                cwd=cwd, env=env)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print('  TIMEOUT after %.0fs' % elapsed)
        return elapsed, False
    elapsed = time.time() - t0

    success = result.returncode == 0
    if not success:
        print('  FAILED (returncode %d)' % result.returncode)
        # Print last lines of combined output for debugging
        output = (result.stdout + result.stderr).strip()
        for line in output.split('\n')[-10:]:
            print('    ' + line)
    return elapsed, success


def run_sampler(bench_dir, sampler, replicate, parallel=None, overrides=None,
                timeout=14400):
    """
    Run one replicate of a sampler. Adjusts the output_dir in a temporary
    copy of the .conf so each replicate writes to its own directory.
    overrides: dict of config key -> value to override (e.g. {'max_iterations': 200})
    """
    conf_src = os.path.join(bench_dir, '%s.conf' % sampler)
    if not os.path.isfile(conf_src):
        print('  Skipping %s (no .conf file)' % sampler)
        return None

    overrides = overrides or {}
    run_output_dir = os.path.join(bench_dir, 'runs', '%s_rep%d' % (sampler, replicate))

    # Create a patched .conf with the per-replicate output_dir
    conf_patched = os.path.join(run_output_dir, '%s.conf' % sampler)
    os.makedirs(run_output_dir, exist_ok=True)

    override_keys = set(overrides.keys()) | {'output_dir'}
    with open(conf_src) as f:
        lines = f.readlines()
    with open(conf_patched, 'w') as f:
        for line in lines:
            key = line.strip().split('=')[0].strip() if '=' in line else ''
            if key == 'output_dir':
                f.write('output_dir = %s/output/\n' % run_output_dir)
            elif key in overrides:
                f.write('%s = %s\n' % (key, overrides[key]))
            else:
                f.write(line)

    print('  %s rep %d ...' % (sampler, replicate), end=' ', flush=True)
    elapsed, success = run_single(conf_patched, cwd=bench_dir, parallel=parallel,
                                   timeout=timeout)

    # Validate that the run actually produced samples (guards against silent
    # early termination, e.g. job pool exhaustion from all-out-of-bounds proposals)
    if success:
        samples_file = os.path.join(run_output_dir, 'output', 'Results', 'samples.txt')
        if os.path.isfile(samples_file):
            with open(samples_file) as f:
                n_lines = sum(1 for _ in f)
            if n_lines <= 1:  # header only, no data
                print('EMPTY (no samples produced in %.1fs)' % elapsed)
                success = False
            else:
                print('%.1fs' % elapsed)
        else:
            print('EMPTY (no samples.txt in %.1fs)' % elapsed)
            success = False
    return {'sampler': sampler, 'replicate': replicate, 'wall_clock': elapsed,
            'success': success, 'output_dir': run_output_dir + '/output'}


# ---------------------------------------------------------------------------
# Parsing results
# ---------------------------------------------------------------------------

def parse_diagnostics(output_dir):
    """Parse diagnostics.txt and return the final row as a dict."""
    diag_file = os.path.join(output_dir, 'Results', 'diagnostics.txt')
    if not os.path.isfile(diag_file):
        return None
    with open(diag_file) as f:
        header_line = f.readline().lstrip('# ').strip()
        headers = header_line.split('\t')
        last_line = None
        all_lines = []
        for line in f:
            line = line.strip()
            if line:
                last_line = line
                all_lines.append(line)
    if last_line is None:
        return None

    vals = last_line.split('\t')
    result = {}
    for h, v in zip(headers, vals):
        try:
            result[h] = float(v)
        except ValueError:
            result[h] = float('nan')

    # Also extract the full trajectory for plotting
    trajectory = {h: [] for h in headers}
    for line in all_lines:
        vals = line.split('\t')
        for h, v in zip(headers, vals):
            try:
                trajectory[h].append(float(v))
            except ValueError:
                trajectory[h].append(float('nan'))
    result['_trajectory'] = trajectory
    return result


def parse_samples(output_dir):
    """Parse samples.txt and return parameter samples as a dict of arrays."""
    samples_file = os.path.join(output_dir, 'Results', 'samples.txt')
    if not os.path.isfile(samples_file):
        return None
    with open(samples_file) as f:
        header = f.readline().lstrip('# ').strip().split('\t')
    data = np.genfromtxt(samples_file, delimiter='\t', skip_header=1,
                         usecols=range(1, len(header)))
    if data.ndim < 2 or data.shape[0] == 0:
        return None
    # header[0] = Name, header[1] = Ln_probability, header[2:] = param names
    result = {'ln_probability': data[:, 0]}
    for i, name in enumerate(header[2:]):
        result[name] = data[:, i + 1]
    return result


def extract_metrics(run_info, ground_truth):
    """Extract all metrics from a single run."""
    odir = run_info['output_dir']
    diag = parse_diagnostics(odir)
    samples = parse_samples(odir)

    metrics = {
        'sampler': run_info['sampler'],
        'replicate': run_info['replicate'],
        'wall_clock': run_info['wall_clock'],
        'success': run_info['success'],
    }

    if diag is not None:
        # Extract per-parameter R-hat, bulk ESS, tail ESS from final diagnostic row
        param_names = [k.replace('rhat_', '') for k in diag if k.startswith('rhat_')]
        metrics['param_names'] = param_names

        rhat_vals = [diag.get('rhat_%s' % p, float('nan')) for p in param_names]
        bulk_ess_vals = [diag.get('bulk_ess_%s' % p, float('nan')) for p in param_names]
        tail_ess_vals = [diag.get('tail_ess_%s' % p, float('nan')) for p in param_names]

        metrics['max_rhat'] = np.nanmax(rhat_vals) if rhat_vals else float('nan')
        metrics['min_bulk_ess'] = np.nanmin(bulk_ess_vals) if bulk_ess_vals else float('nan')
        metrics['min_tail_ess'] = np.nanmin(tail_ess_vals) if tail_ess_vals else float('nan')
        metrics['bulk_ess'] = np.array(bulk_ess_vals)
        metrics['tail_ess'] = np.array(tail_ess_vals)

        total_evals = diag.get('total_evaluations', float('nan'))
        metrics['total_evaluations'] = total_evals
        if total_evals > 0:
            metrics['min_ess_per_eval'] = np.nanmin(bulk_ess_vals) / total_evals
        else:
            metrics['min_ess_per_eval'] = float('nan')

        metrics['_trajectory'] = diag['_trajectory']

    if samples is not None:
        n_samples = len(samples.get('ln_probability', []))
        metrics['n_samples'] = n_samples

        # Compute distance metric D if ground truth is available
        if ground_truth and 'posterior_mean' in ground_truth:
            param_names = metrics.get('param_names', [])
            true_mean = np.array(ground_truth['posterior_mean'])
            true_var = np.array(ground_truth['posterior_variance'])
            true_std = np.sqrt(true_var)
            d = len(true_mean)

            sampled_vals = np.column_stack([samples[p] for p in param_names])
            sampled_mean = np.mean(sampled_vals, axis=0)
            sampled_std = np.std(sampled_vals, axis=0, ddof=1)

            # Laloy & Vrugt (2012) Eq. 10:
            # D = sqrt(1/d * sum[(mu_hat - mu)^2/sigma^2 + (s_hat - sigma)^2/sigma^2])
            D = np.sqrt(np.mean(
                (sampled_mean - true_mean) ** 2 / true_var +
                (sampled_std - true_std) ** 2 / true_var
            ))
            metrics['distance_D'] = D
            metrics['sampled_mean'] = sampled_mean
            metrics['sampled_std'] = sampled_std

    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(all_metrics):
    """Print a formatted comparison table across samplers."""
    by_sampler = {}
    for m in all_metrics:
        by_sampler.setdefault(m['sampler'], []).append(m)

    print('\n' + '=' * 90)
    print('BENCHMARK COMPARISON TABLE')
    print('=' * 90)

    header = '%-8s %5s %10s %10s %10s %10s %10s %10s' % (
        'Sampler', 'N', 'Wall(s)', 'MaxR-hat', 'MinBulkESS', 'ESS/eval', 'D_metric', 'Samples')
    print(header)
    print('-' * 90)

    for sampler in SAMPLERS:
        runs = by_sampler.get(sampler, [])
        successful = [r for r in runs if r.get('success')]
        if not successful:
            print('%-8s  no successful runs' % sampler)
            continue

        n = len(successful)

        def fmt_mean_std(key):
            vals = [r[key] for r in successful if key in r and np.isfinite(r[key])]
            if not vals:
                return '%10s' % 'N/A'
            return '%5.2f+-%4.2f' % (np.mean(vals), np.std(vals))

        def fmt_mean_std_int(key):
            vals = [r[key] for r in successful if key in r and np.isfinite(r[key])]
            if not vals:
                return '%10s' % 'N/A'
            return '%6.0f+-%4.0f' % (np.mean(vals), np.std(vals))

        wall = [r['wall_clock'] for r in successful]
        wall_str = '%5.1f+-%4.1f' % (np.mean(wall), np.std(wall))

        rhat_str = fmt_mean_std('max_rhat')
        ess_str = fmt_mean_std_int('min_bulk_ess')

        ess_eval_vals = [r['min_ess_per_eval'] for r in successful
                         if 'min_ess_per_eval' in r and np.isfinite(r['min_ess_per_eval'])]
        ess_eval_str = '%6.4f+-%4.4f' % (np.mean(ess_eval_vals), np.std(ess_eval_vals)) if ess_eval_vals else '%10s' % 'N/A'

        d_vals = [r['distance_D'] for r in successful if 'distance_D' in r and np.isfinite(r['distance_D'])]
        d_str = '%5.3f+-%4.3f' % (np.mean(d_vals), np.std(d_vals)) if d_vals else '%10s' % 'N/A'

        samp_str = fmt_mean_std_int('n_samples')

        print('%-8s %5d %10s %10s %10s %10s %10s %10s' % (
            sampler, n, wall_str, rhat_str, ess_str, ess_eval_str, d_str, samp_str))

    print('=' * 90)


def save_results_json(all_metrics, output_path):
    """Save all metrics to a JSON file for post-hoc analysis."""
    serializable = []
    for m in all_metrics:
        entry = {}
        for k, v in m.items():
            if k.startswith('_'):
                continue
            if isinstance(v, np.ndarray):
                entry[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                entry[k] = float(v)
            else:
                entry[k] = v
        serializable.append(entry)
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print('Results saved to %s' % output_path)


def plot_convergence(all_metrics, output_path):
    """Plot R-hat and ESS convergence trajectories per sampler."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plots')
        return

    by_sampler = {}
    for m in all_metrics:
        if m.get('success') and '_trajectory' in m:
            by_sampler.setdefault(m['sampler'], []).append(m['_trajectory'])

    if not by_sampler:
        print('No trajectory data available for plotting')
        return

    colors = {'am': '#1f77b4', 'dream': '#ff7f0e', 'p_dream': '#d62728'}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # R-hat trajectory
    ax = axes[0]
    for sampler, trajs in by_sampler.items():
        # Find R-hat columns and take the max across parameters at each step
        for traj in trajs:
            rhat_cols = [k for k in traj if k.startswith('rhat_')]
            if not rhat_cols:
                continue
            iters = np.array(traj['iteration'])
            max_rhat = np.nanmax([traj[c] for c in rhat_cols], axis=0)
            ax.plot(iters, max_rhat, color=colors.get(sampler, 'gray'),
                    alpha=0.4, linewidth=0.8)
        # Plot mean line
        if trajs:
            rhat_cols = [k for k in trajs[0] if k.startswith('rhat_')]
            if rhat_cols:
                all_rhat = []
                min_len = min(len(t['iteration']) for t in trajs)
                for t in trajs:
                    all_rhat.append(np.nanmax([t[c][:min_len] for c in rhat_cols], axis=0))
                mean_rhat = np.nanmean(all_rhat, axis=0)
                iters = np.array(trajs[0]['iteration'][:min_len])
                ax.plot(iters, mean_rhat, color=colors.get(sampler, 'gray'),
                        linewidth=2, label=sampler)
    ax.axhline(y=1.05, color='red', linestyle='--', alpha=0.5, label='threshold')
    ax.set_ylabel('Max R-hat')
    ax.set_ylim(bottom=0.95)
    ax.legend()
    ax.set_title('Convergence Diagnostics')

    # ESS trajectory (min bulk ESS)
    ax = axes[1]
    for sampler, trajs in by_sampler.items():
        for traj in trajs:
            ess_cols = [k for k in traj if k.startswith('bulk_ess_')]
            if not ess_cols:
                continue
            iters = np.array(traj['iteration'])
            min_ess = np.nanmin([traj[c] for c in ess_cols], axis=0)
            ax.plot(iters, min_ess, color=colors.get(sampler, 'gray'),
                    alpha=0.4, linewidth=0.8)
        if trajs:
            ess_cols = [k for k in trajs[0] if k.startswith('bulk_ess_')]
            if ess_cols:
                all_ess = []
                min_len = min(len(t['iteration']) for t in trajs)
                for t in trajs:
                    all_ess.append(np.nanmin([t[c][:min_len] for c in ess_cols], axis=0))
                mean_ess = np.nanmean(all_ess, axis=0)
                iters = np.array(trajs[0]['iteration'][:min_len])
                ax.plot(iters, mean_ess, color=colors.get(sampler, 'gray'),
                        linewidth=2, label=sampler)
    ax.set_ylabel('Min Bulk ESS')
    ax.set_xlabel('Iteration')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print('Convergence plot saved to %s' % output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run MCMC sampler comparison benchmark')
    parser.add_argument('benchmark_dir', help='Path to benchmark directory (e.g., gaussian_d5)')
    parser.add_argument('--replicates', '-n', type=int, default=5,
                        help='Number of replicates per sampler (default: 5)')
    parser.add_argument('--parallel', '-p', type=int, default=None,
                        help='Number of parallel workers for PyBNF')
    parser.add_argument('--samplers', '-s', nargs='+', default=SAMPLERS,
                        help='Which samplers to run (default: am dream p_dream)')
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Override max_iterations in all configs')
    parser.add_argument('--burn-in', type=int, default=None,
                        help='Override burn_in in all configs')
    parser.add_argument('--timeout', type=int, default=14400,
                        help='Per-run timeout in seconds (default: 14400 = 4h)')
    args = parser.parse_args()

    bench_dir = os.path.abspath(args.benchmark_dir)
    if not os.path.isdir(bench_dir):
        print('Error: %s is not a directory' % bench_dir)
        sys.exit(1)

    # Load ground truth if available
    gt_path = os.path.join(bench_dir, 'ground_truth.json')
    ground_truth = None
    if os.path.isfile(gt_path):
        with open(gt_path) as f:
            ground_truth = json.load(f)
        print('Loaded ground truth: %s' % ground_truth.get('description', ''))

    print('\nRunning %d replicates for samplers: %s' % (args.replicates, ', '.join(args.samplers)))
    print('Benchmark directory: %s\n' % bench_dir)

    # Build config overrides from CLI args
    overrides = {}
    if args.max_iterations is not None:
        overrides['max_iterations'] = args.max_iterations
    if args.burn_in is not None:
        overrides['burn_in'] = args.burn_in

    # Run all replicates (retry failed runs up to max_retries times)
    max_retries = 3
    run_results = []
    for sampler in args.samplers:
        for rep in range(args.replicates):
            for attempt in range(1, max_retries + 1):
                result = run_sampler(bench_dir, sampler, rep, args.parallel,
                                     overrides=overrides, timeout=args.timeout)
                if result is None:
                    break  # missing .conf, no point retrying
                if result.get('success'):
                    break
                if attempt < max_retries:
                    print('  %s rep %d failed (attempt %d/%d), retrying...'
                          % (sampler, rep, attempt, max_retries))
            if result is not None:
                run_results.append(result)

    # Collect metrics
    print('\nCollecting metrics...')
    all_metrics = []
    for run_info in run_results:
        if run_info['success']:
            metrics = extract_metrics(run_info, ground_truth)
            all_metrics.append(metrics)

    if not all_metrics:
        print('No successful runs to analyze.')
        sys.exit(1)

    # Report
    print_comparison_table(all_metrics)

    # Save results
    results_dir = os.path.join(bench_dir, 'runs')
    os.makedirs(results_dir, exist_ok=True)
    save_results_json(all_metrics, os.path.join(results_dir, 'results.json'))
    plot_convergence(all_metrics, os.path.join(results_dir, 'convergence.png'))


if __name__ == '__main__':
    main()
