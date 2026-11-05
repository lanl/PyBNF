"""Jobs for the Liu–West filter's tests: an SIR model whose cumulative infections are reported at
a fraction ``rho`` through ``cases()``, daily counts, and an edition-2 configuration."""

import logging
import os
import sys
from pathlib import Path

import numpy as np

from pybnf.algorithms.filters.liu_west import LiuWestFilter
from pybnf.budget import FitBudget
from pybnf.parse import load_config

MODEL = """begin model
begin parameters
  beta   0.6
  gamma  0.25
  N      10000
  I0     10
  rho    0.5
  beta_N beta/N
end parameters
begin molecule types
  S()
  I()
  R()
  C()
end molecule types
begin seed species
  S() N-I0
  I() I0
  R() 0
  C() 0
end seed species
begin observables
  Molecules Inf I()
  Molecules CumInf C()
end observables
begin functions
  cases() rho*CumInf
end functions
begin reaction rules
  S() + I() -> I() + I() + C()  beta_N
  I() -> R()  gamma
end reaction rules
end model
"""

#: Negative-binomial counts (r = 20) around the model's daily increments, days 1 to 12.
COUNTS = (4, 6, 5, 10, 14, 23, 28, 38, 42, 71, 70, 156)

BASE = (
    'edition = 2',
    'model: {model}',
    'bngl_backend = bngsim',
    'job_type = lwf',
    'output_dir = {out}',
    'experiment: epi, data: {data}',
    'noise_model = neg_bin, dispersion = fit r, location = mean',
    'noise_model cases = neg_bin, dispersion = fit r, location = mean, cumulative',
    'loguniform_var = beta 0.2 2',
    'uniform_var = gamma 0.1 0.5',
    'loguniform_var = r 1 100',
    'loguniform_var = I0 1 100',
    'lwf_particles = 200',
    'lwf_forecast_intervals = 2',
    'random_seed = 1',
    'verbosity = 0',
)


def write_data(path, counts=COUNTS, times=None, header='cases'):
    times = np.arange(1, len(counts) + 1) if times is None else times
    with open(path, 'w') as f:
        f.write('# time %s\n' % header + ''.join('%g %s\n' % (t, c) for t, c in zip(times, counts)))
    return str(path)


def write_job(folder, replace=None, extra=(), counts=COUNTS, times=None, header='cases',
              model=MODEL):
    """Write the model, data and configuration under ``folder``; return the conf path.
    ``replace`` maps the start of a base line to its replacement, or to None to drop it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'sir.bngl').write_text(model)
    data = write_data(folder / 'cases.exp', counts, times, header)
    lines = []
    for line in BASE:
        line = line.format(model=folder / 'sir.bngl', out=folder / 'out', data=data)
        line = next((new for start, new in (replace or {}).items() if line.startswith(start)),
                    line)
        if line is not None:
            lines.append(line)
    conf = folder / 'lwf.conf'
    conf.write_text('\n'.join(lines + list(extra)) + '\n')
    return str(conf)


def build(conf_path):
    """The filter, built in this working directory (model initialization changes it)."""
    config = load_config(conf_path)
    os.makedirs(config.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        return LiuWestFilter(config)
    finally:
        os.chdir(home)


def run(alg, client=None):
    """Make the folders main() makes and run the filter; return its output folder."""
    os.makedirs(alg.res_dir, exist_ok=True)
    os.makedirs(alg.sim_dir, exist_ok=True)
    home = os.getcwd()
    try:
        alg.run(client)
    finally:
        os.chdir(home)
    return Path(alg.res_dir) / 'LWF'


def read_table(path):
    """``(header, rows)`` of one of the filter's tables."""
    with open(path) as f:
        lines = [line for line in f if not line.startswith('#')]
    header = lines[0].rstrip('\n').split('\t')
    rows = np.loadtxt(lines[1:], delimiter='\t', ndmin=2) if len(lines) > 1 else np.zeros((0, len(header)))
    return header, rows


def run_main(monkeypatch, folder, conf, *flags, overwrite=True):
    """``pybnf -c conf [-o] -L none *flags`` through main() in ``folder``; its exit code. What
    main() changes for the session (verbosity, sim registry, root logger, ~/dask-worker-space)
    is put back or kept from happening."""
    import pybnf.printing as printing
    from pybnf import pybnf as pybnf_main
    monkeypatch.chdir(folder)
    monkeypatch.setattr(sys, 'argv', ['pybnf', '-c', str(conf), *(['-o'] if overwrite else []),
                                      '-L', 'none', *flags])
    monkeypatch.setattr(printing, 'verbosity', printing.verbosity)
    monkeypatch.setenv('PYBNF_SIM_REGISTRY', '')
    monkeypatch.setattr(pybnf_main, '_cleanup_dask_workspace', lambda: None)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        pybnf_main.main()
    except SystemExit as exc:
        return exc.code
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
    raise AssertionError('main() returned without exiting')


class Draws:
    """A generator stand-in whose normal draws are the given array."""

    def __init__(self, z):
        self.z = z

    def standard_normal(self, shape):
        assert shape == self.z.shape
        return self.z


def unit_draws(x, w, rng, orthogonal=True, d=None):
    """``d`` columns (``x``'s by default) of draws with weighted mean exactly 0, weighted covariance
    exactly ``I`` and, when ``orthogonal``, weighted cross-covariance exactly 0 with every column
    of ``x``."""
    n = x.shape[0]
    basis = np.column_stack([np.ones(n), x]) if orthogonal else np.ones((n, 1))
    g = rng.standard_normal((n, x.shape[1] if d is None else d))
    g -= basis @ np.linalg.solve(basis.T @ (basis * w[:, None]), basis.T @ (g * w[:, None]))
    return g @ np.linalg.inv(np.linalg.cholesky(g.T @ (g * w[:, None]))).T


def one_second_per_row(alg, limit):
    """A budget of ``limit`` seconds on a clock that advances one second per assimilated row."""
    now = [0.0]
    alg.budget = FitBudget(limit, clock=lambda: now[0])
    update = alg._update

    def update_and_tick(pop, k):
        update(pop, k)
        now[0] += 1.0

    alg._update = update_and_tick


def files(folder):
    """Every file under ``folder``, by its path relative to it, as bytes."""
    folder = Path(folder)
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob('*'))
            if p.is_file()}
