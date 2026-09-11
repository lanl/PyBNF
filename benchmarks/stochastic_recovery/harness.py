"""The PyBNF-dependent half of the stochastic recovery benchmark.

Generates a problem's data, builds a fit of it by one of the baseline methods, runs
the fit inline (no dask cluster) while counting every simulation, and records what
the fit reported. The scoring itself is :mod:`protocol`.

Everything runs through the real bngsim backend: BNG2.pl expands the rules once
when the fit is built, and every stochastic trajectory after that is a real SSA or
NFsim run. Only the dask layer is replaced, by the same synchronous doubles the
recovery test tier uses (``tests/integration_harness.py``); they are copied here
rather than imported so this directory depends on nothing outside the ``pybnf``
package and can move to another repository as a unit.

Seeds
-----

Under PyBNF's default ``stochastic_seed = auto`` policy a trajectory's seed is
derived from the parameter values and a replicate index, so a fit that evaluates
the true parameter set draws replicates 0, 1, 2, ... of exactly the process the
data were drawn from. The data are therefore drawn at replicate indices starting
at the problem's ``seed_offset`` (a million), far past any index a fit reaches, so
no fit ever reproduces a data trajectory instead of drawing a fresh one.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from pybnf import config as pybnf_config
from pybnf.algorithms import Result, core
from pybnf.pset import PSet
from pybnf.registry import FIT_TYPE_REGISTRY

from .protocol import Problem, log10_errors, score_fit

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Baseline methods
# --------------------------------------------------------------------------- #
#: The methods the baseline scores, as the conf keys that select and shape them.
#: Each pair (``ss`` / ``ss_noise``, ``cmaes`` / ``cmaes_noise``) differs only in
#: the noise-handling toggle, so a difference between them is that feature's effect.
#: ``max_iterations`` is set past any budget so the simulation budget, enforced by
#: the runner, is the binding stop for every method alike; DE's convergence stop is
#: turned off for the same reason (a noisy population never converges anyway).
METHODS = {
    'de': {'fit_type': 'de', 'population_size': 20, 'stop_tolerance': 0},
    'ss': {'fit_type': 'ss', 'population_size': 10, 'reserve_size': 200, 'ss_noise_handling': 0},
    'ss_noise': {'fit_type': 'ss', 'population_size': 10, 'reserve_size': 200, 'ss_noise_handling': 1},
    'cmaes': {'fit_type': 'cmaes', 'population_size': 12, 'cmaes_noise_handling': 0},
    'cmaes_noise': {'fit_type': 'cmaes', 'population_size': 12, 'cmaes_noise_handling': 1},
}

_UNBOUNDED_ITERATIONS = 10 ** 6

#: How many of a fit's top parameter sets are run again at the end, and how many
#: times each, to decide which is really best (#659). Part of the protocol: the
#: answer a stochastic fit reports is the confirmed one, and the simulations the
#: confirmation spends count against the fit.
BEST_FIT_CANDIDATES = 10
BEST_FIT_REPLICATES = 10


# --------------------------------------------------------------------------- #
# Inline client (mirrors tests/integration_harness.py)
# --------------------------------------------------------------------------- #
class _Future:
    """A future that runs its callable the first time its result is asked for.

    Lazy on purpose: the run loop submits a whole generation of jobs at once and
    consumes their results one at a time, so an eager double would have run the
    entire generation before the runner's budget check could stop anything. Running
    a job when its result is consumed lets the budget stop within one evaluation.
    """

    def __init__(self, fn=None, args=(), value=None):
        self._fn = fn
        self._args = args
        self._value = value
        self.status = 'finished'

    def result(self):
        if self._fn is not None:
            self._value = self._fn(*self._args)
            self._fn = None
        return self._value


class InlineClient:
    """Runs every submitted callable in this process, when its result is consumed, and
    returns a finished future. ``cluster`` is set so the run reports itself as a local
    run rather than a cluster one."""
    cluster = 'local'

    def scatter(self, objs, broadcast=False):
        return [_Future(value=o) for o in objs]

    def submit(self, fn, *args, **kwargs):
        return _Future(fn, args)

    def cancel(self, futures):
        pass

    def scheduler_info(self):
        return {'workers': {}}


class _AsCompleted:
    """Synchronous ``as_completed``: yields ``(future, result)`` in submission order
    and accepts ``update()`` so the run loop can enqueue resubmissions."""

    def __init__(self, futures, with_results=False, raise_errors=True, timeout=None):
        assert with_results and not raise_errors
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


class SimulationCounter:
    """Counts every job the run loop executes: one job is one simulation of one
    model, under ``smoothing`` too, since ``make_job`` makes one job per replicate."""

    def __init__(self):
        self.n = 0

    def run_job(self, j, debug=False, failed_logs_dir=''):
        self.n += 1
        return _slim_run_job(j)


def _slim_run_job(j):
    """Folder-free stand-in for ``core.run_job``: run the models, score on the worker
    when the fit scattered its objective (it does not under smoothing, where the
    master averages the replicates first and scores the average)."""
    simdata = j._run_models()
    res = Result(j.params, simdata, j.job_id)
    if j.calc_future is not None:
        res.normalize(j.norm_settings)
        res.score = j.calc_future.result().evaluate_objective(res.simdata, res.pset, show_warnings=False)
        res.out = simdata
        if res.score is None:
            res.score = np.inf
    return res


class _BudgetedDecision:
    """Wraps ``Algorithm._record_result_and_decide`` to trace the fit's reported best
    parameter set and to stop the run once the simulation budget is spent.

    A module-level callable rather than a closure because the run loop pickles the
    algorithm for its periodic backup, and a closure stored on the instance would make
    that pickle fail.
    """

    def __init__(self, alg, counter, problem, budget, trace):
        self.alg = alg
        self.counter = counter
        self.truth = problem.truth
        self.names = problem.names
        self.budget = budget
        self.trace = trace
        self.best_name = None
        self.search_simulations = None
        self._real = alg._record_result_and_decide

    def __call__(self, res):
        alg = self.alg
        decision = self._real(res)
        if len(alg.trajectory):
            name = alg.trajectory.best_fit_name()
            if name != self.best_name:
                self.best_name = name
                self.trace.append((self.counter.n,
                                   log10_errors(best_values(alg, self.names), self.truth)))
        if decision == 'STOP':
            self.search_simulations = self.counter.n
        elif self.counter.n >= self.budget:
            self.search_simulations = self.counter.n
            alg.stop_reason = ('Simulation budget reached: stopped after %d simulation(s)'
                               % self.counter.n)
            return 'STOP'
        return decision


class _InlineRun:
    """Context manager that routes the run loop through the inline doubles and a
    simulation counter, restoring the real dask seam afterwards."""

    def __init__(self):
        self.counter = SimulationCounter()

    def __enter__(self):
        self._saved = (core.as_completed, core.run_job)
        core.as_completed = _AsCompleted
        core.run_job = self.counter.run_job
        return self

    def __exit__(self, *exc):
        core.as_completed, core.run_job = self._saved
        return False


# --------------------------------------------------------------------------- #
# Building a fit
# --------------------------------------------------------------------------- #
def _placeholder_exp(problem: Problem, path: Path):
    cols = list(problem.observables) + [o + '_SD' for o in problem.observables]
    path.write_text('#\ttime\t' + '\t'.join(cols) + '\n0\t' + '\t'.join('1' for _ in cols) + '\n')


def make_config(problem: Problem, workdir, *, fit_type='de', exp_path=None, objfunc='chi_sq',
                seed=1234, smoothing=None, **overrides):
    """A legacy-syntax PyBNF Configuration for a fit of ``problem``.

    Every free parameter is a ``loguniform_var`` over its frozen bounds. The data
    file defaults to the problem's committed data; ``exp_path`` overrides it (data
    generation passes a placeholder, since it never scores anything).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    model_path = str(problem.model_path)
    exp_path = str(exp_path if exp_path is not None else problem.data_path)
    base = {
        'models': {model_path}, model_path: [exp_path], 'exp_data': {exp_path},
        'output_dir': str(workdir / 'out'),
        'objfunc': objfunc, 'fit_type': fit_type,
        'bngl_backend': 'bngsim',
        'initialization': 'lh',
        'delete_old_files': 1,
        'verbosity': 0,
        'wall_time_sim': 0,
        'random_seed': int(seed),
        'smoothing': int(problem.smoothing if smoothing is None else smoothing),
        'max_iterations': _UNBOUNDED_ITERATIONS,
        'population_size': 20,
        'num_to_output': 1000,
        'output_every': 10 ** 6,
        'best_fit_candidates': BEST_FIT_CANDIDATES,
        'best_fit_replicates': BEST_FIT_REPLICATES,
    }
    for p in problem.parameters:
        base[('loguniform_var', p.name)] = [p.low, p.high]
    base.update(overrides)
    return pybnf_config.Configuration(base)


def build(conf):
    """Construct the algorithm the configuration names. This runs BNG2.pl network
    generation (for a network-based method) and the bngsim conversion; it chdir-s,
    so the cwd is restored afterwards."""
    os.makedirs(conf.config['output_dir'], exist_ok=True)
    home = os.getcwd()
    try:
        return FIT_TYPE_REGISTRY[conf.config['fit_type']].cls(conf)
    finally:
        os.chdir(home)


def pset_for(alg, values):
    """A PSet of the algorithm's free variables at ``values`` (``{name: value}``)."""
    return PSet([v.set_value(values[v.name]) for v in alg.variables])


def best_values(alg, names):
    best = alg.trajectory.best_fit()
    return {name: float(best[name]) for name in names}


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def simulate_replicates(problem: Problem, workdir, values, n, first_index):
    """``n`` trajectories of the model at ``values``, at replicate indices
    ``first_index, first_index + 1, ...``. Returns ``(times, array[n, T, n_obs])``
    with the observables in the problem's order."""
    workdir = Path(workdir)
    placeholder = workdir / (problem.suffix + '.exp')
    workdir.mkdir(parents=True, exist_ok=True)
    _placeholder_exp(problem, placeholder)
    conf = make_config(problem, workdir, exp_path=placeholder, objfunc='sos',
                       population_size=4, max_iterations=1, smoothing=1)
    alg = build(conf)
    model = alg.model_list[0].copy_with_param_set(pset_for(alg, values))
    folder = workdir / 'sims'
    folder.mkdir(exist_ok=True)
    home = os.getcwd()
    runs = []
    times = None
    try:
        for i in range(n):
            model._pybnf_replicate_index = first_index + i
            ds = model.execute(str(folder), 'rep_%d' % i, 0)
            data = ds[problem.suffix]
            arr = np.asarray(data.data)
            if times is None:
                times = arr[:, data.cols[data.indvar]].copy()
            runs.append(np.column_stack([arr[:, data.cols[o]] for o in problem.observables]))
    finally:
        os.chdir(home)
    return times, np.asarray(runs)


def write_exp(path, times, observables, mean, sigma):
    """Write the data file: the independent variable, each observable's replicate
    mean, and each observable's ``_SD`` column (the per-point sigma ``chi_sq`` reads)."""
    cols = ['time'] + list(observables) + [o + '_SD' for o in observables]
    lines = ['#\t' + '\t'.join(cols)]
    for k in range(len(times)):
        row = [times[k]] + list(mean[k]) + list(sigma[k])
        lines.append('\t'.join('%.10g' % v for v in row))
    Path(path).write_text('\n'.join(lines) + '\n')


def generate_data(problem: Problem, workdir=None, out_path=None):
    """Draw the problem's data at its true parameter values and write the data file.

    The data are the mean over ``data_replicates`` trajectories at each sampling
    time, drawn at replicate indices ``seed_offset ..``; the sigma column is the
    standard deviation across those trajectories, floored at ``sd_floor_fraction``
    of the observable's peak mean so a point every trajectory agrees on (the initial
    condition, a species that stays at zero) does not get infinite weight.

    Returns the path written. The file is deterministic given the definition and
    the simulator, and it is committed, so the benchmark does not depend on the
    simulator's random stream staying the same.
    """
    tmp = None
    if workdir is None:
        tmp = workdir = tempfile.mkdtemp(prefix='srb_' + problem.id[:8] + '_')
    try:
        times, runs = simulate_replicates(problem, workdir, problem.truth,
                                          problem.data_replicates, problem.data_seed_offset)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    mean = runs.mean(axis=0)
    sd = runs.std(axis=0, ddof=1)
    floor = problem.sd_floor_fraction * np.abs(mean).max(axis=0)
    sigma = np.maximum(sd, floor[None, :])
    out = Path(out_path) if out_path is not None else problem.data_path
    write_exp(out, times, problem.observables, mean, sigma)
    return out


def read_exp(path):
    """Columns of a data file as ``{name: array}`` (the header names, in order)."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    header = lines[0].lstrip('#').split()
    arr = np.array([[float(x) for x in ln.split()] for ln in lines[1:]])
    return {name: arr[:, i] for i, name in enumerate(header)}


# --------------------------------------------------------------------------- #
# Scoring a parameter set the way a fit would
# --------------------------------------------------------------------------- #
def objective_at(alg, values, replicate_offset=0):
    """The objective value a fit of ``alg`` would record for ``values``: the mean of
    ``smoothing`` replicates scored as one evaluation, exactly the run loop's path."""
    pset = pset_for(alg, values)
    jobs = alg.make_job(pset, replicate_offset=replicate_offset)
    folded = None
    for job in jobs:
        res = _slim_run_job(job)
        if len(jobs) == 1:
            folded = res
        else:
            folded = alg._fold_group_result(res)
    return float(alg.score_result(folded))


def leverage(problem: Problem, workdir=None, n_truth=6, factors=(0.5, 2.0), seed=3):
    """How much a factor-of-two change in each parameter moves the frozen objective,
    in units of the objective's own noise at the truth.

    Scores the truth ``n_truth`` times at distinct replicate offsets to get the mean
    and standard deviation of the objective there, then each parameter alone at each
    factor. Returns ``{'truth_mean', 'truth_sd', 'z': {name: [z per factor]}}``. A
    parameter whose z is small in both directions is one the data barely see at the
    baseline replicate count; one whose z is far below its siblings' is a narrow
    direction the optimizer has to find.
    """
    tmp = None
    if workdir is None:
        tmp = workdir = tempfile.mkdtemp(prefix='srb_lev_' + problem.id[:8] + '_')
    try:
        alg = build(make_config(problem, workdir, fit_type='de', seed=seed))
        truth = problem.truth
        smoothing = int(problem.smoothing)
        at_truth = [objective_at(alg, truth, replicate_offset=k * smoothing) for k in range(n_truth)]
        mean = float(np.mean(at_truth))
        sd = float(np.std(at_truth, ddof=1))
        z = {}
        for name in problem.names:
            row = []
            for f in factors:
                values = dict(truth)
                values[name] = truth[name] * f
                row.append((objective_at(alg, values) - mean) / sd)
            z[name] = row
        return {'truth_mean': mean, 'truth_sd': sd, 'factors': list(factors), 'z': z}
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Running one fit
# --------------------------------------------------------------------------- #
def run_fit(problem: Problem, method, seed, workdir=None, budget=None, keep_workdir=False,
            overrides=None, label=None):
    """Fit ``problem`` with baseline ``method`` from fit seed ``seed`` and return its
    scored record (see :func:`protocol.score_fit`).

    ``overrides`` are conf keys laid over the method's own (``{'ss_noise_max_draws': 2}``),
    which is how a variant of a baseline method is scored without editing ``METHODS``;
    ``label`` is the method name the record carries for such a variant (default: ``method``).

    The fit stops when it has spent ``budget`` simulations (default: the problem's
    frozen budget), then runs PyBNF's end-of-fit confirmation of the best fit (#659),
    which decides the answer and whose simulations count too; the record's
    ``simulations`` is everything the fit ran and ``simulations_search`` the count when
    the search itself stopped. The trace holds the fit's reported best parameter set
    each time it changed, as ``(simulations spent, {parameter: error})``.
    """
    if method not in METHODS:
        raise KeyError('unknown method %r (have %s)' % (method, sorted(METHODS)))
    budget = int(problem.budget_simulations if budget is None else budget)
    tmp = None
    if workdir is None:
        tmp = workdir = tempfile.mkdtemp(prefix='srb_%s_%s_%d_' % (problem.id[:8], method, seed))
    spec = dict(METHODS[method])
    spec.update(overrides or {})
    fit_type = spec.pop('fit_type')
    started = time.time()
    try:
        conf = make_config(problem, workdir, fit_type=fit_type, seed=seed, **spec)
        alg = build(conf)
        truth = problem.truth
        trace = []

        with _InlineRun() as inline:
            counter = inline.counter
            decide = _BudgetedDecision(alg, counter, problem, budget, trace)
            alg._record_result_and_decide = decide
            os.makedirs(alg.sim_dir, exist_ok=True)
            os.makedirs(alg.res_dir, exist_ok=True)
            home = os.getcwd()
            try:
                alg.run(InlineClient())
            finally:
                os.chdir(home)
            simulations = counter.n

        estimate = best_values(alg, problem.names) if len(alg.trajectory) else {}
        trace.append((simulations, log10_errors(estimate, truth)))
        record = score_fit(problem, estimate, simulations, trace,
                           method=label or method, seed=int(seed), budget=budget,
                           base_method=method, overrides=dict(overrides or {}),
                           simulations_search=(decide.search_simulations
                                               if decide.search_simulations is not None
                                               else simulations),
                           best_score=(float(alg.trajectory.best_score()) if len(alg.trajectory)
                                       else None),
                           wall_time=time.time() - started,
                           stop_reason=getattr(alg, 'stop_reason', None) or 'algorithm stop')
        return record
    finally:
        if tmp is not None and not keep_workdir:
            shutil.rmtree(tmp, ignore_errors=True)


def run_fit_json(args):
    """``run_fit`` for a process pool: ``(problem_dir, method, seed, budget)`` or
    ``(problem_dir, method, seed, budget, overrides, label)`` in, the JSON-serializable
    record out."""
    from .protocol import load_problem
    problem_dir, method, seed, budget = args[:4]
    overrides, label = (args[4], args[5]) if len(args) > 4 else (None, None)
    return run_fit(load_problem(problem_dir), method, seed, budget=budget, overrides=overrides, label=label)


__all__ = ['METHODS', 'BEST_FIT_CANDIDATES', 'BEST_FIT_REPLICATES', 'InlineClient',
           'SimulationCounter', 'make_config', 'build', 'pset_for', 'best_values',
           'simulate_replicates', 'write_exp', 'generate_data', 'read_exp', 'objective_at',
           'leverage', 'run_fit', 'run_fit_json']
