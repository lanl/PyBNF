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
    """Return (cols, arr): the model's output at the true parameters through the real
    bngsim backend, for the dataset's protocol -- a plain time course, a steady-state
    parameter scan (``doses``), or a two-phase pre-equilibration (``condition`` /
    ``preequilibrate``). For the two-phase case we read the MEASUREMENT suffix
    (``timecourse``), not the unmeasured equilibration phase (``timecourse_preequil``)."""
    H.require_bng2pl()
    folder = example.path
    model_file = dataset.model or example.model   # a dataset may name its own model (joint fit)
    indvar = dataset.scan or 'time'
    if dataset.doses:
        grid = list(dataset.doses)            # the .exp rows ARE the swept doses (no time grid)
    else:
        grid = list(np.linspace(0, dataset.t_end, dataset.n_points))
    # Everything the fit machinery writes (network gen, output_dir, truth run)
    # goes to a scratch dir so the committed example folder stays pristine.
    scratch = Path(tempfile.mkdtemp(prefix='tutorial_gen_'))
    try:
        placeholder = scratch / dataset.exp
        placeholder.write_text('#\t' + indvar + '\t' + '\t'.join(dataset.obs) + '\n' +
                               '\n'.join(f'{x:.10g}\t' + '\t'.join('0' for _ in dataset.obs)
                                         for x in grid) + '\n')
        extra = {}
        if dataset.condition is not None:
            extra['condition'] = dataset.condition
        if dataset.preequilibrate is not None:
            extra['preequilibrate'] = dataset.preequilibrate
        conf = H.make_newera_config(scratch, str(folder / model_file), placeholder,
                                    example.build_free, 'timecourse', 'de',
                                    population_size=4, max_iterations=1, **extra)
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
        data = ds['timecourse'] if 'timecourse' in ds else ds[next(iter(ds))]
        return data.cols, np.asarray(data.data)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _write_measurement_exp(path, cols, arr, dataset):
    """Write a ``.exp`` of DERIVED measurement-model columns (ADR-0036).

    The raw model observables in ``dataset.obs`` were just simulated; here we
    materialize each ``dataset.measurements`` formula over them -- via the very
    same :class:`~pybnf.measurement.MeasurementModel` the fit uses -- with any
    observation-layer nuisance at its true value, and write those columns (not the
    raw observables). Generating the data with the fit's own layer code is what
    makes the recovery exact.
    """
    from pybnf.data import Data
    from pybnf.measurement import MeasurementModel

    names = [name for name, _ in sorted(cols.items(), key=lambda kv: kv[1])]
    raw = Data.from_columns(np.asarray(arr, dtype=float), names)
    time = raw['time']
    header = ['time']
    columns = [time]
    for m in dataset.measurements:
        allowed = set(dataset.obs) | set(m.nuisance)
        col = MeasurementModel(m.obs_id, m.formula, allowed).materialize(raw, dict(m.nuisance))
        header.append(m.obs_id)
        columns.append(np.asarray(col, dtype=float))
    lines = ['# ' + '\t'.join(header)]
    for row in np.column_stack(columns):
        lines.append('\t'.join('%.10g' % v for v in row))
    Path(path).write_text('\n'.join(lines) + '\n')


def _negative_binomial_counts(means, dispersion, seed):
    """Draw over-dispersed integer counts: negative-binomial samples with the given
    per-point ``means`` and dispersion ``r`` (the NB2 parameterization, so
    variance = mean + mean**2/r). numpy's ``negative_binomial(n, p)`` has mean
    n(1-p)/p, so n = r and p = r/(r+mean) reproduce that mean exactly; a zero mean
    (an empty cell at t=0) deterministically yields a zero count."""
    rng = np.random.default_rng(seed)
    means = np.asarray(means, dtype=float)
    p = dispersion / (dispersion + np.where(means > 0, means, 0.0))
    out = np.zeros(means.shape, dtype=float)
    nonzero = means > 0
    out[nonzero] = rng.negative_binomial(dispersion, p[nonzero])
    return out


def _write_exp(path, cols, arr, obs, noise_sd=0.0, noise_seed=0, sd=None, outliers=(),
               indvar='time', count_dispersion=None, count_seed=0, scale=None):
    """Write the simulated trajectory as a ``.exp``.

    ``indvar`` is the independent-variable column (``time`` for a time course, or the
    swept parameter name for a dose-response scan). Beyond the obs columns, three
    optional corruptions support the noise/robust lessons: gaussian ``noise_sd``
    (added to every point, seeded), explicit ``outliers`` -- ``(row_index,
    obs_name, replacement_value)`` triples spliced into the named observable column
    (deterministic gross errors), and ``count_dispersion`` -- resample every
    observable as over-dispersed integer COUNTS (negative-binomial, mean = the model
    value, seeded by ``count_seed``) for the count-likelihood lesson. A ``_SD`` column
    is written per observable when either ``sd`` (a constant, independent of the
    gaussian noise) or ``noise_sd`` is set, so chi_sq / laplace have a per-point scale
    to weight by; count data carries no ``_SD`` (a count likelihood is self-normalizing).
    """
    idx = [cols[indvar]] + [cols[o] for o in obs]
    header = [indvar] + list(obs)
    rows = arr[:, idx].copy()
    if scale is not None:
        rows[:, 1:] *= scale        # observable columns in arbitrary units (scale-free objective)
    if count_dispersion is not None:
        for j in range(1, rows.shape[1]):
            rows[:, j] = _negative_binomial_counts(
                rows[:, j], count_dispersion, count_seed + j)
    if noise_sd > 0:
        rng = np.random.default_rng(noise_seed)
        for j in range(1, rows.shape[1]):
            rows[:, j] = rows[:, j] + rng.normal(0.0, noise_sd, size=rows.shape[0])
    for row_index, obs_name, value in outliers:
        rows[row_index, 1 + list(obs).index(obs_name)] = value   # col 0 is the indvar
    sd_value = sd if sd is not None else (noise_sd if noise_sd > 0 else None)
    if sd_value is not None:
        header += [o + '_SD' for o in obs]
        rows = np.hstack([rows, np.full((rows.shape[0], len(obs)), sd_value)])
    lines = ['# ' + '\t'.join(header)]
    for row in rows:
        fmt = '%d' if count_dispersion is not None else '%.10g'
        lines.append('\t'.join(
            ('%.10g' % v if k == 0 else fmt % v) for k, v in enumerate(row)))
    Path(path).write_text('\n'.join(lines) + '\n')


def regenerate(example):
    for dataset in example.datasets:
        cols, arr = _simulate_truth(example, dataset)
        if dataset.measurements:
            _write_measurement_exp(example.path / dataset.exp, cols, arr, dataset)
            derived = ', '.join(m.obs_id for m in dataset.measurements)
            print(f'  wrote {example.folder}/{dataset.exp}  '
                  f'[{len(arr)} pts, measurement-model: {derived}]')
        else:
            _write_exp(example.path / dataset.exp, cols, arr, dataset.obs,
                       dataset.noise_sd, dataset.noise_seed, dataset.sd, dataset.outliers,
                       indvar=dataset.scan or 'time',
                       count_dispersion=dataset.count_dispersion, count_seed=dataset.count_seed,
                       scale=dataset.scale)
            tag = f' (+N({dataset.noise_sd}) seed {dataset.noise_seed})' if dataset.noise_sd else ''
            if dataset.scale is not None:
                tag += f' (scaled x{dataset.scale:g}, arbitrary units)'
            if dataset.count_dispersion is not None:
                tag += f' (neg-bin counts, r={dataset.count_dispersion}, seed {dataset.count_seed})'
            if dataset.outliers:
                tag += f' (+{len(dataset.outliers)} outliers, _SD={dataset.sd})'
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
