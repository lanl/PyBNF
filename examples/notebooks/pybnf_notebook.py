"""Tiny helpers for driving PyBNF (and its bngsim backend) interactively from a
Jupyter notebook.

Why this exists
---------------
PyBNF's command-line entry, ``pybnf -c fit.conf``, spins up a Dask *distributed*
cluster and farms each simulation out to worker **processes**. That is the right
design for a large production fit on a workstation or an HPC allocation. Inside a
notebook you almost always want the opposite: a single-process, **synchronous**
run whose Python objects -- the fitted model, the posterior samples -- stay live
in the kernel so you can poke at them in the next cell.

``run_fit`` gives you exactly that. It drives PyBNF's *real* algorithm classes
(the same ``DifferentialEvolution`` / ``TrustRegionReflective`` / ``HMC`` code the
CLI runs) to completion inline, using a drop-in synchronous stand-in for the Dask
client. No cluster is started, nothing is pickled across a process boundary, and
the finished :class:`~pybnf.algorithms.base.Algorithm` is handed back to you. This
is the same in-process execution path PyBNF's own test suite uses to check every
tutorial lesson (``tests/integration_harness.py``), so it exercises the production
code, not a reimplementation of it.

The public surface is deliberately small:

* :func:`run_fit`         -- build + run a fit from an edition-2 ``.conf`` string.
* :func:`build_algorithm` -- build the Algorithm *without* running it (e.g. to reach
  into ``alg.model_list`` for a forward simulation).
* :func:`best_fit`        -- the winning parameter values as a plain ``dict``.
* :func:`simulate`        -- run one model at chosen parameters and return the
  observable trajectory as a :class:`pandas.DataFrame`.
* :func:`load_exp`        -- read a PyBNF ``.exp`` data file into a DataFrame.
* :func:`net_path` / :func:`bngsim_model` -- reach the BNG2.pl-generated ``.net`` and
  a fresh :class:`bngsim.Model` for pure-bngsim forward simulation.

Everything here needs ``BNGPATH`` pointed at a BioNetGen install for any BNGL model
(network generation shells out to ``BNG2.pl`` once); analytical/``expression``
targets (lesson 37/38, notebook 03) need no simulator at all.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from pybnf import config, parse
import pybnf.algorithms.core as _core
from pybnf.pset import PSet
from pybnf.registry import FIT_TYPE_REGISTRY


# --------------------------------------------------------------------------- #
# A synchronous, in-process stand-in for the Dask distributed client.
#
# PyBNF's run loop only touches four bits of the client surface: ``scatter`` /
# ``submit`` / ``cancel`` on the client, and ``as_completed(..., with_results=True,
# raise_errors=False)`` with an ``.update()`` method to enqueue resubmissions. We
# provide exactly those, running every task the instant it is submitted, in this
# process. No cluster, no serialization -- so the models (including the bngsim
# engine handles that do not survive a pickle round-trip) stay put and results are
# immediate.
# --------------------------------------------------------------------------- #
class _Future:
    def __init__(self, result):
        self._result = result
        self.status = 'finished'

    def result(self):
        return self._result


class _SyncClient:
    def scatter(self, objs, broadcast=False):
        return [_Future(o) for o in objs]

    def submit(self, fn, *args, **kwargs):
        return _Future(fn(*args))

    def cancel(self, futures):
        pass


class _SyncAsCompleted:
    """Synchronous ``as_completed``: yields ``(future, future.result())`` and
    supports ``update()`` so the run loop can enqueue resubmitted jobs."""

    def __init__(self, futures, with_results=False, raise_errors=True):
        self._queue = list(futures)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._queue:
            raise StopIteration
        f = self._queue.pop(0)
        return f, f.result()

    def update(self, new_futures):
        self._queue.extend(new_futures)


# --------------------------------------------------------------------------- #
# Building + running
# --------------------------------------------------------------------------- #
def build_algorithm(conf_text, output_dir):
    """Parse an edition-2 ``.conf`` string and build (but do not run) the fit.

    ``output_dir`` is created fresh (any existing one is removed, so a notebook
    cell is safe to re-run). For a BNGL model this is where the constructor runs
    BNG2.pl network generation, writing ``<output_dir>/Initialize/<model>_gen_net.net``.

    Returns ``(alg, conf)`` -- the real :class:`~pybnf.algorithms.base.Algorithm`
    subclass selected by ``job_type`` and its :class:`~pybnf.config.Configuration`.
    """
    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    raw = parse.ploop(conf_text.splitlines(keepends=True))
    raw['output_dir'] = str(output_dir)
    raw.setdefault('verbosity', 0)
    conf = config.Configuration(raw)
    # `job_type` (edition-2) is normalized to the internal `fit_type` registry key.
    alg = FIT_TYPE_REGISTRY[conf.config['fit_type']].cls(conf)
    return alg, conf


def run_fit(conf_text, output_dir):
    """Build and run a fit fully in-process; return the finished Algorithm.

    The run is synchronous and single-process (see the module docstring). Read the
    result off the returned object: ``best_fit(alg)`` for the winning parameters,
    ``alg.res_dir`` for the ``Results/`` directory PyBNF wrote (best-fit model,
    sorted parameter tables, and -- for a sampler -- ``samples.txt`` /
    ``inference_data.nc``).
    """
    alg, _ = build_algorithm(conf_text, output_dir)
    os.makedirs(alg.sim_dir, exist_ok=True)
    os.makedirs(alg.res_dir, exist_ok=True)

    saved = _core.as_completed
    _core.as_completed = _SyncAsCompleted   # the run loop calls core.as_completed
    try:
        alg.run(_SyncClient())
    finally:
        _core.as_completed = saved
    return alg


# --------------------------------------------------------------------------- #
# Reading results + forward simulation
# --------------------------------------------------------------------------- #
def best_fit(alg):
    """The winning (lowest-objective) parameter values as a ``{name: value}`` dict."""
    bf = alg.trajectory.best_fit()
    return {v.name: float(bf[v.name]) for v in alg.variables}


def _data_to_frame(data):
    """A PyBNF ``Data`` object (``.cols`` name->index, ``.data`` array) -> DataFrame."""
    names = [n for n, _ in sorted(data.cols.items(), key=lambda kv: kv[1])]
    return pd.DataFrame(np.asarray(data.data, dtype=float), columns=names)


def simulate(alg, params=None, model_index=0):
    """Run one of the fit's models at ``params`` and return its trajectory.

    ``params`` is a ``{name: value}`` dict over the fit's free parameters; ``None``
    uses the current best fit. The model is simulated on the experiment's own output
    grid (the data times). Returns a :class:`pandas.DataFrame` (a ``time`` column
    plus one column per model observable) when the model has a single simulation
    action, else a ``{suffix: DataFrame}`` dict.
    """
    values = params if params is not None else best_fit(alg)
    model = alg.model_list[model_index]
    pset = PSet([v.set_value(values[v.name]) for v in alg.variables])
    scratch = Path(alg.sim_dir) / 'notebook_sim'
    scratch.mkdir(parents=True, exist_ok=True)
    ds = model.copy_with_param_set(pset).execute(str(scratch), 'nb', 0)
    frames = {suffix: _data_to_frame(d) for suffix, d in ds.items()}
    return next(iter(frames.values())) if len(frames) == 1 else frames


def net_path(alg, model_index=0):
    """Path to the BNG2.pl-generated ``.net`` for a BNGL model in the fit."""
    return alg.model_list[model_index]._net_path


def bngsim_model(alg, model_index=0):
    """A **fresh** :class:`bngsim.Model` loaded from the fit's generated ``.net``.

    Fresh (not the fit's live engine model) so you can ``set_param`` and re-run it
    for a parameter sweep without disturbing anything -- start each sweep point from
    a new model so its initial state is the model's own seed species."""
    import bngsim
    return bngsim.Model.from_net(net_path(alg, model_index))


def _first_parameter(bngl_text):
    """The name of the first entry in the model's ``begin parameters`` block."""
    in_block = False
    for line in bngl_text.splitlines():
        s = line.split('#', 1)[0].strip()
        if not s:
            continue
        if s.startswith('begin parameters'):
            in_block = True
            continue
        if in_block:
            if s.startswith('end parameters'):
                break
            return s.split()[0]
    raise ValueError('no begin parameters block found in the model')


def bngl_to_net(bngl_path, work_dir):
    """Expand a BNGL model's rules into a flat reaction network and return the
    ``.net`` path.

    Shells out to ``BNG2.pl`` once (the same network generation PyBNF runs at the
    start of every BNGL fit) via a throwaway single-parameter config. Needs
    ``BNGPATH``. Hand the returned path to :func:`bngsim.Model.from_net` -- reloading
    from the ``.net`` for each parameter-sweep point gives you a fresh model whose
    initial state is the model's own seed species.
    """
    bngl_path = Path(bngl_path)
    dummy = _first_parameter(bngl_path.read_text())   # a throwaway free var so the conf validates
    placeholder = Path(work_dir) / '_placeholder.exp'
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text('# time\tt\n0\t0\n1\t0\n2\t0\n')
    conf_text = (
        'edition = 2\n'
        f'model: {bngl_path}\n'
        'bngl_backend = bngsim\n'
        'job_type = de\n'
        'objective = sos\n'
        f'experiment: gen, data: {placeholder}\n'
        f'uniform_var = {dummy} 0 1\n'
        'population_size = 3\n'
        'max_iterations = 1\n'
    )
    alg, _ = build_algorithm(conf_text, Path(work_dir) / '_netgen')
    return net_path(alg)


def compile_bngl(bngl_path, work_dir):
    """Generate the reaction network for a BNGL model and return a ready
    :class:`bngsim.Model` (a convenience wrapper over :func:`bngl_to_net`).

    Use the returned model with :class:`bngsim.Simulator` for pure forward
    simulation -- no fitting involved. Needs ``BNGPATH``."""
    import bngsim
    return bngsim.Model.from_net(bngl_to_net(bngl_path, work_dir))


def load_exp(path):
    """Read a PyBNF ``.exp`` (or PEtab-derived ``.exp``) file into a DataFrame.

    The first line is a ``# col col ...`` header; columns are whitespace/tab
    separated. The independent variable is usually ``time``."""
    path = Path(path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    header = lines[0].lstrip('#').split()
    rows = [ln.split() for ln in lines[1:] if not ln.lstrip().startswith('#')]
    arr = np.array(rows, dtype=float)
    return pd.DataFrame(arr, columns=header)
