#!/usr/bin/env python3
"""Run the stochastic recovery benchmark and summarize its results.

Subcommands::

    run        fit problems x methods x seeds, appending one record per fit to a
               JSON results file (resumable: pairs already in the file are skipped);
               --set key=value --as name scores a variant of a method under its own name
    summarize  aggregate a results file into the per-(problem, method) table
    generate   regenerate a problem's data file from its frozen definition
    leverage   how far a factor-of-two change in each parameter moves the objective,
               in units of the objective's noise at the truth

Examples::

    python benchmarks/stochastic_recovery/run_baseline.py run \\
        --seeds 5 --parallel 8 --out benchmarks/stochastic_recovery/results/baseline_v1.json
    python benchmarks/stochastic_recovery/run_baseline.py summarize \\
        benchmarks/stochastic_recovery/results/baseline_v1.json
    python benchmarks/stochastic_recovery/run_baseline.py generate --check

Needs bngsim and BNG2.pl (set ``BNGPATH``); the network-free problem also needs
bngsim's NFsim backend. Fits run in worker processes, each through the real
simulator in-process, so ``--parallel`` should not exceed the machine's cores.
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from stochastic_recovery import protocol  # noqa: E402


def _load_records(path):
    path = Path(path)
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def _save_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(records, indent=1) + '\n')
    tmp.replace(path)


def _parse_setting(text):
    """``key=value`` from the command line, the value read as an int, a float, or a string."""
    if '=' not in text:
        sys.exit('--set expects key=value, got %r' % text)
    key, value = text.split('=', 1)
    for convert in (int, float):
        try:
            return key.strip(), convert(value)
        except ValueError:
            pass
    return key.strip(), value


def cmd_run(args):
    from stochastic_recovery.harness import METHODS, run_fit_json
    problems = protocol.load_problems(ids=args.problems or None)
    methods = args.methods or list(METHODS)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        sys.exit('unknown method(s): %s (have %s)' % (unknown, sorted(METHODS)))
    overrides = dict(_parse_setting(s) for s in (args.set or []))
    if args.label and len(methods) != 1:
        sys.exit('--as names one method variant; pass exactly one --methods with it')
    if overrides and not args.label:
        sys.exit('--set changes a method, so name the variant with --as (a record must say what ran)')
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    records = _load_records(args.out)
    done = {(r['problem'], r['method'], r['seed']) for r in records}
    jobs = []
    # The heavy (network-free) fits go first, so they do not trail the run at the end.
    for p in sorted(problems, key=lambda p: p.method != 'nf'):
        budget = (int(round(p.budget_simulations * args.budget_scale))
                  if args.budget is None else args.budget)
        for m in methods:
            name = args.label or m
            for s in seeds:
                if (p.id, name, s) not in done:
                    jobs.append((str(p.directory), m, s, budget, overrides, args.label))
    print('%d fit(s) to run (%d already in %s), %d worker(s)'
          % (len(jobs), len(done), args.out, args.parallel), flush=True)
    if not jobs:
        return
    started = time.time()
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(args.parallel) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_fit_json, jobs), 1):
            records.append(rec)
            _save_records(args.out, records)
            print('[%3d/%d %6.0fs] %-46s %-12s seed %d  sims %6d  max err %.3f  %s'
                  % (i, len(jobs), time.time() - started, rec['problem'], rec['method'], rec['seed'],
                     rec['simulations'], rec['max_error'],
                     'ok' if rec['success_loose'] else '--'), flush=True)
    print('done in %.0f s' % (time.time() - started))


def cmd_summarize(args):
    records = _load_records(args.results)
    rows = protocol.aggregate(records)
    table = protocol.format_table(rows)
    print(table)
    if args.out:
        Path(args.out).write_text(table + '\n')


def cmd_generate(args):
    from stochastic_recovery.harness import generate_data, read_exp
    import numpy as np
    problems = protocol.load_problems(ids=args.problems or None)
    for p in problems:
        if args.check:
            tmp = p.directory / (p.suffix + '.regenerated.exp')
            generate_data(p, out_path=tmp)
            old, new = read_exp(p.data_path), read_exp(tmp)
            same = (list(old) == list(new)
                    and all(np.allclose(old[k], new[k], rtol=0, atol=0) for k in old))
            tmp.unlink()
            print('%s: regenerated data %s the committed file' % (p.id, 'MATCHES' if same else 'DIFFERS FROM'))
        elif p.data_path.is_file() and not args.force:
            print('%s: %s exists; pass --force to overwrite' % (p.id, p.data_path.name))
        else:
            out = generate_data(p)
            print('%s: wrote %s' % (p.id, out))


def cmd_leverage(args):
    from stochastic_recovery.harness import leverage
    print('| problem | parameter | halved | doubled |')
    print('|---|---|---:|---:|')
    for p in protocol.load_problems(ids=args.problems or None):
        lev = leverage(p)
        for name, (lo, hi) in lev['z'].items():
            print('| %s | `%s` | %d | %d |' % (p.id, name[:-6], round(lo), round(hi)), flush=True)
        print('<!-- %s: objective at the truth %.1f +- %.1f -->'
              % (p.id, lev['truth_mean'], lev['truth_sd']), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    r = sub.add_parser('run', help='run fits')
    r.add_argument('--problems', nargs='*', help='problem ids (default: all)')
    r.add_argument('--methods', nargs='*', help='baseline method names (default: all)')
    r.add_argument('--seeds', type=int, default=5, help='fit seeds per (problem, method)')
    r.add_argument('--first-seed', type=int, default=1)
    r.add_argument('--budget', type=int, default=None, help='override every problem\'s simulation budget')
    r.add_argument('--budget-scale', type=float, default=1.0, help='scale every problem\'s budget')
    r.add_argument('--set', action='append', metavar='KEY=VALUE',
                   help='lay a conf key over the method (repeatable); requires --as')
    r.add_argument('--as', dest='label', help='record the fits under this method name (a variant of --methods)')
    r.add_argument('--parallel', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    r.add_argument('--out', default=str(_HERE / 'results' / 'latest.json'))
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser('summarize', help='aggregate a results file')
    s.add_argument('results')
    s.add_argument('--out', help='also write the Markdown table here')
    s.set_defaults(fn=cmd_summarize)

    g = sub.add_parser('generate', help='(re)generate data files')
    g.add_argument('--problems', nargs='*')
    g.add_argument('--force', action='store_true')
    g.add_argument('--check', action='store_true',
                   help='regenerate to a temporary file and report whether it matches the committed one')
    g.set_defaults(fn=cmd_generate)

    lv = sub.add_parser('leverage', help='how much a factor-of-two change in each parameter moves the objective')
    lv.add_argument('--problems', nargs='*')
    lv.set_defaults(fn=cmd_leverage)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
