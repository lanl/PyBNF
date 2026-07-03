#!/usr/bin/env python
"""Regenerate the committed ``.exp`` data files for the edition-2 tutorial.

This is a **developer tool**, not part of the test run. For each example in
``_manifest.py`` it simulates the committed model at its known-true parameters
through the real bngsim backend and writes the observable trajectory as a
zero-noise ``.exp`` (plus a seeded-noise ``_SD`` variant where the manifest asks
for one). The data is nothing more than the model's own output at the truth --
there is no hidden transformation of the model.

Usage (needs bngsim + BNG2.pl; set BNGPATH):

    python examples/tutorial/regenerate_data.py            # all examples
    python examples/tutorial/regenerate_data.py 02_bateman_chain   # one folder
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import tests.recovery_harness as H          # noqa: E402
from tests.integration_harness import FakeAsCompleted, slim_run_job  # noqa: E402
import pybnf.algorithms as _alg             # noqa: E402
from pybnf.pset import PSet                 # noqa: E402

from _manifest import EXAMPLES              # noqa: E402

# Run the fit machinery inline (no dask cluster) -- the simulation is still real.
_alg.core.as_completed = FakeAsCompleted
_alg.core.run_job = slim_run_job


def _simulate_truth(example, dataset):
    """Return (cols, arr): the model's trajectory at the true parameters, on the
    dataset's time grid, through the real bngsim backend."""
    H.require_bng2pl()
    folder = example.path
    grid = np.linspace(0, dataset.t_end, dataset.n_points)
    # Everything the fit machinery writes (network gen, output_dir, truth run)
    # goes to a scratch dir so the committed example folder stays pristine.
    scratch = Path(tempfile.mkdtemp(prefix='tutorial_gen_'))
    try:
        placeholder = scratch / dataset.exp
        placeholder.write_text('#\ttime\t' + '\t'.join(dataset.obs) + '\n' +
                               '\n'.join(f'{t:.10g}\t' + '\t'.join('0' for _ in dataset.obs)
                                         for t in grid) + '\n')
        conf = H.make_newera_config(scratch, str(folder / example.model), placeholder,
                                    example.build_free, 'timecourse', 'de',
                                    population_size=4, max_iterations=1)
        alg = H.build(conf, 'de')
        truth = PSet([v.set_value(example.truth[v.name]) for v in alg.variables])
        model = alg.model_list[0].copy_with_param_set(truth)
        out = str(scratch / 'truth')
        os.makedirs(out, exist_ok=True)
        home = os.getcwd()
        try:
            ds = model.execute(out, 'truth', 0)
        finally:
            os.chdir(home)
        data = ds[next(iter(ds))]
        return data.cols, np.asarray(data.data)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _write_exp(path, cols, arr, obs, noise_sd=0.0, noise_seed=0):
    idx = [cols['time']] + [cols[o] for o in obs]
    header = ['time'] + list(obs)
    rows = arr[:, idx].copy()
    if noise_sd > 0:
        rng = np.random.default_rng(noise_seed)
        for j in range(1, rows.shape[1]):
            rows[:, j] = rows[:, j] + rng.normal(0.0, noise_sd, size=rows.shape[0])
        header += [o + '_SD' for o in obs]
        sd_block = np.full((rows.shape[0], len(obs)), noise_sd)
        rows = np.hstack([rows, sd_block])
    lines = ['# ' + '\t'.join(header)]
    for row in rows:
        lines.append('\t'.join('%.10g' % v for v in row))
    Path(path).write_text('\n'.join(lines) + '\n')


def regenerate(example):
    for dataset in example.datasets:
        cols, arr = _simulate_truth(example, dataset)
        _write_exp(example.path / dataset.exp, cols, arr, dataset.obs,
                   dataset.noise_sd, dataset.noise_seed)
        tag = f' (+N({dataset.noise_sd}) seed {dataset.noise_seed})' if dataset.noise_sd else ''
        print(f'  wrote {example.folder}/{dataset.exp}  '
              f'[{len(arr)} pts, obs {dataset.obs}]{tag}')


def main(argv):
    wanted = set(argv[1:])
    for example in EXAMPLES:
        if wanted and example.folder not in wanted:
            continue
        print(example.folder)
        regenerate(example)


if __name__ == '__main__':
    main(sys.argv)
