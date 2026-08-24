"""Shared harness for fast, in-process integration tests of the fitting
algorithms against analytical targets.

The idea
--------
``AnalyticalModel`` (``pybnf/analytical_model.py``) evaluates a closed-form
negative-log-likelihood (Gaussian / banana / multimodal) with **no simulation
backend**, and ``objfunc = direct_pass`` feeds that NLL straight to the
optimizer/sampler. So we can drive the *real* algorithm classes
(``DifferentialEvolution``, ``Adaptive_MCMC``, ``DreamAlgorithm``, ...) end to
end against a target whose true optimum / posterior moments are known in closed
form — the analytical gold standard.

Two things are faked, and only two, so these stay *integration* tests of the
algorithms rather than of dask/the filesystem:

  * **dask** — a synchronous ``FakeClient`` + ``FakeAsCompleted`` (same public
    surface ``Algorithm.run`` depends on: ``scatter``/``submit``/``cancel`` and
    ``as_completed(..., with_results=True, raise_errors=False)`` with
    ``.update()``). Jobs run inline; no cluster, no subprocess.
  * **the per-evaluation simulation folder** — ``slim_run_job`` runs the model
    and scores it without ``mkdir``-ing a folder per evaluation (the dominant
    orchestration cost; ``AnalyticalModel.execute`` ignores the folder anyway).
    Real folder/file end-to-end behaviour is already covered by
    ``test_run_loop.py`` / ``test_job_execution.py``.

Everything else is the real code path: ``start_run`` → proposal/mutation →
``run_job`` → objective → ``got_result`` (acceptance/selection) → trajectory →
convergence checks, plus the sampler's real sample-file output.

What to assert where
--------------------
Full posterior recovery for the MCMC samplers is inherently slow (the samplers
stream per-step output to disk), so:

  * **fast tier** (every change): optimizers must *find the known mode*;
    samplers must satisfy cheap *sanity / directional* invariants on a short run
    (acceptance in a sane band, chain concentrates toward the mode, finite
    diagnostics). These run in seconds.
  * **slow tier** (``@pytest.mark.slow``, opt-in): full moment recovery against
    ``ground_truth`` with statistical tolerances. Run occasionally / before
    landing the critical algorithm patches.
"""
import copy
import json
import os

import numpy as np

from pybnf import algorithms, config
from pybnf.algorithms import Result
from pybnf.data import Data
from pybnf.pset import Model


# --------------------------------------------------------------------------- #
# Synchronous dask substitutes
# --------------------------------------------------------------------------- #
class FakeFuture:
    def __init__(self, result):
        self._result = result
        self.status = 'finished'

    def result(self):
        return self._result


class FakeClient:
    """Runs every submitted callable inline and returns a finished future.

    ``cluster`` is set because this double stands in for a LOCAL run, and a real local
    ``Client`` owns the ``LocalCluster`` it drives. ``_report_parallelism`` reads exactly
    that attribute to tell a local run from a cluster one (#621), so a double without it
    claims to be a cluster run and then fails the ``scheduler_info()`` call that claim
    leads to.

    Nothing breaks: the failure is caught and the fit is unaffected. What it costs is
    attention. ``logger.exception`` records a full traceback, and pytest shows a captured
    log only for a test that FAILS, so the traceback appears in exactly the situation
    where someone is reading the log to find out why something broke, attached to a fit
    that broke for an unrelated reason. It came up while diagnosing #648 and had to be
    ruled out first. The attribute only has to be non-None; nothing reads through it.
    """

    cluster = 'local'

    def scatter(self, objs, broadcast=False):
        return [FakeFuture(o) for o in objs]

    def submit(self, fn, *args, **kwargs):
        return FakeFuture(fn(*args))

    def cancel(self, futures):
        pass


class FakeAsCompleted:
    """Synchronous ``as_completed``: yields ``(future, future.result())`` and
    supports ``update()`` so the run loop can enqueue resubmissions.

    ``timeout`` is accepted (and ignored: everything here completes inline) because
    a fit under a wall-time budget hands the real ``as_completed`` its remaining
    seconds (#529)."""

    def __init__(self, futures, with_results=False, raise_errors=True, timeout=None):
        assert with_results and not raise_errors  # the contract run() relies on
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


def slim_run_job(j, debug=False, failed_logs_dir=''):
    """Folder-free stand-in for ``algorithms.core.run_job``.

    Mirrors the success path of ``Job.run_simulation`` (run models → normalize →
    score via the scattered objective), but skips the per-evaluation ``mkdir``
    and folder teardown, which dominate wall time and are unused by
    ``AnalyticalModel``.
    """
    simdata = j._run_models()
    res = Result(j.params, simdata, j.job_id)
    if j.calc_future is not None:
        res.normalize(j.norm_settings)
        res.score = j.calc_future.result().evaluate_objective(
            res.simdata, res.pset, show_warnings=False)
        res.out = simdata
        if res.score is None:          # NaN/Inf simulation → penalize, don't crash
            res.score = np.inf
    return res


def install(monkeypatch):
    """Patch the dask layer, the simulation-folder layer, and the analytical
    model's dask-race ``sleep`` for the duration of a test."""
    monkeypatch.setattr(algorithms.core, 'as_completed', FakeAsCompleted)
    monkeypatch.setattr(algorithms.core, 'run_job', slim_run_job)
    monkeypatch.setattr('pybnf.analytical_model.time.sleep', lambda *a, **k: None)


# --------------------------------------------------------------------------- #
# Linear-Gaussian forward model — the closed-form DREAM(KZS) oracle (ADR-0067
# Stage 3). Unlike the AnalyticalModel targets (which emit a scalar `score`
# column for `direct_pass`), this emits observable output columns f(x) = A x
# scored by the *real* chi_sq likelihood — the shape the Kalman proposal needs
# (it reads the model output vector, not just the score). With a linear forward
# map, a Gaussian likelihood, and a flat prior, the posterior is Gaussian in
# closed form (`linear_gaussian_posterior`): the honest end-to-end oracle.
# --------------------------------------------------------------------------- #
class LinearGaussianModel(Model):
    """Test-only forward model ``f(x) = A x``: emits one observable row per output
    (``index``, ``y``) scored by chi_sq against a matching ``.exp`` (``index``, ``y``,
    ``y_SD``). Parameters bind by name ``p1..pD`` (``D = A.shape[1]``), matching
    :func:`make_config`; the ``target`` suffix matches the generated ``target.exp``."""

    def __init__(self, A, name='g', pset=None):
        self.A = np.asarray(A, dtype=float)   # n_out x n_params
        self.name = name
        self.file_path = name
        self.suffixes = ['target']
        self.stochastic = False
        self.has_observables = True
        self.param_names = set()              # params come from the config, not the model
        self._pset = pset

    def copy_with_param_set(self, pset):
        m = copy.copy(self)
        m._pset = pset
        return m

    def save(self, file_prefix, **kwargs):
        pass

    def get_suffixes(self):
        return self.suffixes

    def execute(self, folder, filename, timeout):
        if self._pset is None:
            raise ValueError('LinearGaussianModel has no parameter set')
        x = np.array([self._pset['p%d' % (i + 1)] for i in range(self.A.shape[1])])
        y = self.A @ x                        # n_out forward outputs
        arr = np.column_stack([np.arange(len(y), dtype=float), y])
        d = Data(arr=arr)
        d.cols = {'index': 0, 'y': 1}
        d.headers = {0: 'index', 1: 'y'}
        return {'target': d}


def linear_gaussian_posterior(A, d, sigma):
    """Closed-form posterior ``x | d ~ N(mu_post, Sigma_post)`` for the linear-Gaussian
    model with a flat prior: ``Sigma_post = (A^T R^-1 A)^-1``, ``mu_post = Sigma_post A^T
    R^-1 d`` with ``R = diag(sigma**2)``. Returns ``(mu_post, Sigma_post)``."""
    A = np.asarray(A, dtype=float)
    d = np.asarray(d, dtype=float)
    r_inv = np.diag(1.0 / np.asarray(sigma, dtype=float) ** 2)
    prec = A.T @ r_inv @ A
    cov = np.linalg.inv(prec)
    return cov @ A.T @ r_inv @ d, cov


def make_linear_gaussian_config(tmp_path, A, d, sigma, bounds=(-20.0, 20.0), **overrides):
    """A real ``objfunc = chi_sq`` DREAM fit over the linear-Gaussian forward model
    ``f(x) = A x`` scored against ``d`` with per-observation ``sigma``.

    Writes a matching ``target.exp`` (``index``, ``y``, ``y_SD``) and a placeholder
    ``.target`` so the Configuration loader builds a model named ``g`` with suffix
    ``target``; then swaps in :class:`LinearGaussianModel` (keeping name+suffix so the
    loaded exp_data keys still resolve). Parameters are ``p1..pD`` over ``bounds``
    (a wide flat prior, so the Gaussian posterior sits well inside the box)."""
    A = np.asarray(A, dtype=float)
    d = np.asarray(d, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n_out, n_params = A.shape
    tgt = tmp_path / 'g.target'
    tgt.write_text(json.dumps(gaussian_spec([0.0] * n_params, [1.0] * n_params)))
    exp = tmp_path / 'target.exp'
    rows = ['# index\ty\ty_SD']
    rows += ['%d\t%.17g\t%.17g' % (i, d[i], sigma[i]) for i in range(n_out)]
    exp.write_text('\n'.join(rows) + '\n')
    # Sampler defaults so callers that only exercise construction/validation need not
    # restate the required keys; any is overridable via ``overrides``.
    kw = dict(population_size=5, max_iterations=100, burn_in=50)
    kw.update(overrides)
    conf = make_config(tmp_path, 'dream', str(tgt), str(exp), n_params,
                       bounds=bounds, objfunc='chi_sq', **kw)
    conf.models['g'] = LinearGaussianModel(A, name='g')
    return conf


# --------------------------------------------------------------------------- #
# Targets + config
# --------------------------------------------------------------------------- #
def gaussian_spec(mean, variance):
    return {'type': 'gaussian', 'mean': list(mean), 'variance': list(variance)}


def banana_spec(a=1.0, b=100.0):
    return {'type': 'banana', 'a': a, 'b': b}


def multimodal_spec(modes):
    """A Gaussian-mixture target from ``modes`` -- a list of ``(weight, mean,
    variance)`` triples (diagonal variance per mode). NLL is
    ``-logsumexp_k [log w_k - 0.5 (x-mu_k)^2 / var_k]`` (``_nll_multimodal``).

    Well-separated modes are the canonical *honesty* geometry for a local sampler:
    HMC/NUTS follows the gradient into whichever basin it starts in and cannot hop
    the near-zero-density gap between modes, so independent chains park in different
    modes and the cross-chain R-hat stays high -- exactly the failure HMC's own
    diagnostics must flag (ADR-0059)."""
    return {'type': 'multimodal',
            'modes': [{'weight': float(w),
                       'mean': np.asarray(mu, dtype=float).tolist(),
                       'variance': np.asarray(var, dtype=float).tolist()}
                      for w, mu, var in modes]}


def rotated_gaussian_spec(mean, covariance):
    """A correlated/rotated Gaussian target with a full covariance matrix
    ``Sigma`` (NLL ``0.5 (x-mu)^T Sigma^{-1} (x-mu)``; mode = ``mean``). Unlike
    ``gaussian_spec`` (diagonal variance, separable), the off-diagonals tilt the
    bowl's principal axes off the coordinate axes."""
    return {'type': 'rotated_gaussian',
            'mean': np.asarray(mean, dtype=float).tolist(),
            'covariance': np.asarray(covariance, dtype=float).tolist()}


def rotated_quartic_spec(mean, angle, coeff):
    """A smooth, non-separable, NON-quadratic, trap-free valley target:
    ``k1 r1^4 + k2 r2^2`` with ``r = R(angle)(x-mu)`` and ``coeff = (k1, k2)``
    (mode = ``mean``). With ``k1 << k2`` it is a long, flat, curved valley — the
    discriminator for Powell's bracketing+Brent line search, on which the
    fixed-step parabola stalls (#406). Unlike the rotated *Gaussian* (quadratic,
    fit exactly by a parabola), this is genuinely non-quadratic."""
    return {'type': 'rotated_quartic', 'mean': list(mean),
            'angle': float(angle), 'coeff': list(coeff)}


def rotated_cov(variances, angle):
    """Build ``Sigma = R diag(variances) R^T`` (2-D): the principal-axis variances
    ``variances`` rotated by ``angle`` radians. When the two variances differ
    widely the bowl is ill-conditioned, and a non-zero ``angle`` rotates its long
    axis off the coordinate axes -- so coordinate-only descent is slow and the
    target exercises the conjugate-direction / covariance-adaptation machinery."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag(np.asarray(variances, dtype=float)) @ R.T


def write_target(tmp_path, spec):
    """Write a ``.target`` JSON file and the trivial ``target.exp`` it pairs
    with. Returns ``(target_path, exp_path)``. The exp prefix MUST be 'target'
    to match ``AnalyticalModel``'s single suffix."""
    tgt = tmp_path / 'g.target'
    tgt.write_text(json.dumps(spec))
    exp = tmp_path / 'target.exp'
    exp.write_text('# index\tscore\n0\t0\n')
    return str(tgt), str(exp)


def make_config(tmp_path, fit_type, target_path, exp_path, n_params,
                bounds=(-10.0, 10.0), var_type='uniform_var', start=None,
                **overrides):
    """Build a real ``Configuration`` for an analytical fit with parameters
    ``p1..pN``.

    ``var_type='uniform_var'`` (default) gives bounded uniform priors over
    ``bounds`` — used by the population optimizers and the samplers.
    ``var_type='var'`` gives single-value start points (required by Simplex);
    pass ``start=[...]`` for the initial vector."""
    if var_type == 'var':
        if start is None:
            start = [0.0] * n_params
        var_spec = {('var', 'p%d' % (i + 1)): [start[i]] for i in range(n_params)}
    else:
        lo, hi = bounds
        var_spec = {(var_type, 'p%d' % (i + 1)): [lo, hi] for i in range(n_params)}
    base = {
        'output_dir': str(tmp_path) + '/out',
        'models': {target_path}, target_path: [exp_path], 'exp_data': {exp_path},
        'objfunc': 'direct_pass', 'fit_type': fit_type, 'initialization': 'lh',
        'delete_old_files': 1,   # skip run()'s best-fit-copy tail (incompatible
                                 # with AnalyticalModel's string suffixes)
        'verbosity': 0, 'wall_time_sim': 0,
        'random_seed': 1234,     # deterministic by default; override per test
    }
    base.update(var_spec)
    base.update(overrides)
    return config.Configuration(base)


# --------------------------------------------------------------------------- #
# Driving + reading results
# --------------------------------------------------------------------------- #
def drive(alg):
    """Create the output scaffolding ``main()`` normally makes, then run the
    algorithm inline to completion with the fake client.

    No RNG seeding here: the algorithm built its own np.random.Generator from
    ``config['random_seed']`` in its constructor (the legacy global RNG is unused)."""
    os.makedirs(alg.sim_dir, exist_ok=True)
    os.makedirs(alg.res_dir, exist_ok=True)
    alg.run(FakeClient())


def best_params(alg, n_params):
    """Best-fit parameter vector (lowest objective) from the trajectory."""
    bp = alg.trajectory.best_fit()
    return np.array([bp['p%d' % (i + 1)] for i in range(n_params)])


def read_samples(output_dir, n_params):
    """Pooled post-output MCMC samples as an ``(n_samples, n_params)`` array.

    Handles both sampler output conventions:
      * ``Results/samples.txt``                (DREAM / P-DREAM)
      * ``Results/A_MCMC/Runs/params_*.txt``   (Adaptive_MCMC)
    """
    samples_txt = os.path.join(output_dir, 'Results', 'samples.txt')
    if os.path.isfile(samples_txt):
        with open(samples_txt) as f:
            header = f.readline().lstrip('#').split()
        # columns: Name, Ln_probability, <param keys...>
        col = {name: i for i, name in enumerate(header)}
        data = np.genfromtxt(samples_txt, skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] == 0:
            return np.zeros((0, n_params))
        idx = [col['p%d' % (i + 1)] for i in range(n_params)]
        return data[:, idx]

    rows = []
    runs = os.path.join(output_dir, 'Results', 'A_MCMC', 'Runs')
    if os.path.isdir(runs):
        for fn in sorted(os.listdir(runs)):
            if fn.startswith('params_') and fn.endswith('.txt'):
                d = np.genfromtxt(os.path.join(runs, fn), names=True)
                if d.size == 0:
                    continue
                d = np.atleast_1d(d)
                rows.append(np.column_stack([d['p%d' % (i + 1)] for i in range(n_params)]))
    return np.vstack(rows) if rows else np.zeros((0, n_params))


def acceptance_rate(samples_or_array):
    """Fraction of consecutive rows that differ (a proxy for the MCMC
    acceptance rate when reading a pooled, post-thinning sample set)."""
    a = np.asarray(samples_or_array)
    if len(a) < 2:
        return float('nan')
    moved = np.any(np.diff(a, axis=0) != 0, axis=1)
    return float(np.mean(moved))
