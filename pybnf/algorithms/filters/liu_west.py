"""The Liu–West filter (``job_type = lwf``): a bootstrap particle filter whose free parameters
move by the Liu and West (2001) kernel, for forecasting count data that arrive one interval at a
time. The algorithm, its outputs and what it refuses are documented under :ref:`alg-lwf`.

A segment costs far less than a job on the cluster, and an update needs every weight before it
can resample, so, as ``hmc`` and ``ms`` do, this job type overrides
:meth:`~pybnf.algorithms.base.Algorithm.run` and integrates in this process. Unlike ``ms`` it does
not call :meth:`~pybnf.algorithms.base.Algorithm._finalize_run` (ADR-0110): it enters nothing in
the trajectory and has no best fit, so that tail would report a missing best fit for a run that
finished. It writes ``Results/stop_reason.txt`` through the shared tail (ADR-0093); ``main()`` writes
``Results/method_chain.json`` (ADR-0107).
"""

import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import pickle
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import Field

from ... import __version__, _bngsim_caps
from ..base import Algorithm, latin_hypercube
from ...bngsim_model import SegmentFailed, SegmentIntegrator, initial_state_only_ids
from ...budget import FitBudget, format_duration
from ...config_schema import GlobalConfig, PyBNFConfigModel
from ...data import Data
from ...noise import MEAN, ConstantSigma, FreeParameterSigma, NegBinomial
from ...printing import PybnfError, print0, print1, print2
from ...pset import PSet
from ...registry import register_fit_type
from ...run_directories import RUN_FILES, RUN_FOLDERS, SIMULATION_FOLDER

__all__ = ['LWFConfig', 'LiuWestFilter']

logger = logging.getLogger('pybnf.algorithms')

#: The registry family: a filter is neither an optimizer nor a fixed-parameter posterior sampler.
FAMILY = 'filter'

#: What every output file and the run summary say the output is.
SAMPLE_STATEMENT = (
    'This is a forecasting sample of a model whose free parameters drift by the Liu–West '
    'kernel at each row, not the posterior of fixed parameters. For the fixed-parameter '
    'posterior, use an MCMC job type such as am.')

#: ``h`` of a moved parameter with no ``lwf_parameter_jitter`` line when ``lwf_jitter`` is unset.
DEFAULT_JITTER = 0.15

#: The folder under ``Results/`` the filter writes its files to.
OUTPUT_FOLDER = 'LWF'

#: The state file in ``output_dir`` when ``lwf_state_file`` is not set (main() deletes Results/).
DEFAULT_STATE_FILE = 'lwf_state.npz'

#: What a state file says it is, and the version of its layout.
STATE_FORMAT = 'pybnf.lwf.state'
STATE_VERSION = 1

#: The remedy every refusal of a continuation names.
_FRESH_RUN = 'Start a fresh run with lwf_continue = 0, which filters every row from the prior.'

#: The arrays a state file of this version holds.
STATE_KEYS = ('format', 'format_version', 'fingerprint', 'run', 'u', 'state', 'weights',
              'predicted', 'times', 'counts', 'history', 'by_row', 'var_floor')

#: The two noise_model lines that declare the count: edition 2 requires the whole-fit one, and
#: the weights are the column's.
WHOLE_FIT_FORM = 'noise_model = neg_bin, dispersion = {dispersion}, location = mean'
COLUMN_FORM = 'noise_model {column} = neg_bin, dispersion = {dispersion}, location = mean, cumulative'

# Random-stream purposes.
_PRIOR_DRAW, _UPDATE, _FORECAST = 0, 1, 2

# An increment within max(_ZERO_ABS, _ZERO_REL * |cumulative total|) of zero is integrator noise.
_ZERO_ABS = 1e-9
_ZERO_REL = 2.0 ** -42

# The floor on each moved parameter's variance, as a fraction of its prior-draw variance.
_VAR_FLOOR_FRAC = 1e-12

# A boxed parameter's position in its box is kept this far from either wall before its logit.
_EDGE = 1e-12

# Spacing that differs from the step by more than this fraction of it is not one step.
_STEP_TOL = 1e-3

# Distinct particles below this fraction of the population (and at least two) is reported.
_DISTINCT_WARN_FRAC = 0.05


# --------------------------------------------------------------------------------------------
# The configuration surface
# --------------------------------------------------------------------------------------------

class LWFConfig(PyBNFConfigModel):
    """The Liu–West filter's settings (ADR-0006). ``lwf_parameter_jitter`` holds the ``(name, h)``
    pairs of the repeatable ``lwf_parameter_jitter = <name> <h>`` line."""

    lwf_particles: int = Field(4000, ge=2)
    # None applies DEFAULT_JITTER; a set value that no moved parameter uses is refused.
    lwf_jitter: Optional[float] = Field(None, gt=0, lt=1)
    lwf_parameter_jitter: list = Field(default_factory=list)
    lwf_resample_threshold: float = Field(0.5, gt=0, le=1)
    lwf_forecast_intervals: int = Field(4, ge=0)
    lwf_bounds: Literal['logit', 'reflect'] = 'logit'
    lwf_independent_runs: int = Field(1, ge=1)
    lwf_state_file: Optional[str] = None
    lwf_continue: Literal[0, 1] = 0

    @classmethod
    def postprocess(cls, conf_dict, fit_type):
        """Refuse, on the raw configuration, what this job type cannot honour."""
        _require_edition_2(conf_dict)
        _check_keys_not_read(conf_dict)
        _refuse_declarations_not_read(conf_dict)
        _check_one_bngl_model(conf_dict)
        _check_one_experiment(conf_dict)
        _check_independent_runs(conf_dict)
        for name, h in conf_dict.get('lwf_parameter_jitter', ()):
            if not 0.0 < h < 1.0:
                raise PybnfError('lwf_parameter_jitter = %s %r is out of range: h must be strictly '
                                 'between 0 and 1.' % (name, h))
        return conf_dict


# Global keys the filter reads.
_READ_GLOBAL_KEYS = frozenset({
    'edition', 'job_type', 'output_dir', 'random_seed', 'bng_command', 'bngl_backend',
    'generate_network', 'wall_time_fit', 'wall_time_sim', 'wall_time_gen', 'delete_old_files',
    'simulation_dir', 'objective', 'noise_location', 'initialization', 'parallel_count',
    'ind_var_rounding',
})

# Global keys main() acts on, accepted at the one value that asks for nothing.
_REFUSED_VALUES = {'refine': 0, 'bootstrap': 0}

# The cluster main() starts; the filter's independent runs use the local one.
_CLUSTER_KEYS = frozenset({'cluster_type', 'scheduler_file', 'scheduler_node', 'worker_nodes'})

# Global keys the loader acts on before the filter weighs a row, which it cannot honour.
_REFUSED_GLOBAL_KEYS = {
    'normalization': 'the loader would normalize the data before any row is weighed',
    'postprocess': 'it weighs each segment as integrated and runs no postprocessing script',
}

# Keys every configuration carries that the parser or the loader fill in.
_STRUCTURAL_KEYS = frozenset({'fit_type', 'models', 'exp_data', 'verbosity', 'population_size'})

# Declarations (the parser's tuple keys) and experiment: fields the filter does not honour.
_REFUSED_DECLARATIONS = frozenset({
    'start_point', 'normalization', 'time_error', 'condition', 'measurement', 'objective_target',
    'objective_modes', 'model_tolerances'})
_REFUSED_EXPERIMENT_FIELDS = frozenset({
    'condition', 'preequilibrate', 'equil_t_end', 't_start', 't_end', 'n_steps',
    'measurement_params', 'gml', 'complex'})



def _refusal(what, why, hint=None):
    return PybnfError('job_type = lwf does not accept %s: %s.' % (what, why), hint=hint)


def _declaration(column='<obs>', dispersion='fit <r>'):
    """The two noise_model lines, quoted, for a message."""
    return "'%s' and '%s'" % (WHOLE_FIT_FORM.format(dispersion=dispersion),
                              COLUMN_FORM.format(column=column, dispersion=dispersion))


def _dispersion_text(source):
    """``fit <r>`` or ``fix_at <value>``, as the noise_model line writes it."""
    return ('fit %s' % source.name if isinstance(source, FreeParameterSigma)
            else 'fix_at %g' % source.const)


def _require_edition_2(d):
    edition = d.get('edition')
    if not edition or edition < 2:
        raise PybnfError(
            'The Liu–West filter (job_type = lwf) requires the edition-2 configuration '
            'surface, but this configuration is %s.'
            % ('edition 1 (legacy)' if not edition else 'edition %d' % edition),
            hint="Add 'edition = 2', declare the model with 'model:', the data with one "
                 "'experiment:' line, and the count with the two lines %s." % _declaration())


def _check_keys_not_read(d):
    """Refuse a global key main() would act on; report the others the filter does not read as
    unused (ADR-0014). A key no schema owns is the shared check's."""
    own = LWFConfig.owned_keys()
    known = GlobalConfig.owned_keys() | {'max_iterations'}   # structural, and not read here
    for key, why in _REFUSED_GLOBAL_KEYS.items():
        if key in d:
            raise _refusal(key, why, hint='Remove the line.')
    for key, value in d.items():
        if (not isinstance(key, str) or key in own or key in _STRUCTURAL_KEYS
                or key in _READ_GLOBAL_KEYS or key not in known):
            continue
        if key in _REFUSED_VALUES:
            if value != _REFUSED_VALUES[key]:
                raise PybnfError('job_type = lwf accepts %s = %s only.' % (key, _REFUSED_VALUES[key]),
                                 hint='Remove the line.')
        elif key in _CLUSTER_KEYS:
            raise _refusal('%s = %s' % (key, value),
                           'its independent runs use the local cluster main() starts',
                           hint='Remove the line.')
        else:
            print1('Warning: Configuration key %s is not used in job_type lwf, so I am ignoring it'
                   % key)
            logger.warning('Ignoring unused key %s for fitting algorithm lwf', key)
    if d.get('bngl_backend') == 'bionetgen':
        raise _refusal('bngl_backend = bionetgen', 'BNG2.pl cannot start an interval from a '
                       'carried state', hint='Remove the line, or set bngl_backend = bngsim.')


def _refuse_declarations_not_read(d):
    for key in d:
        if isinstance(key, tuple) and key and key[0] in _REFUSED_DECLARATIONS:
            name = '' if key[1] is None else ' for %s' % key[1]
            raise PybnfError('job_type = lwf does not read a %s declaration%s.' % (key[0], name),
                             hint='Remove that line.')


def _check_one_bngl_model(d):
    models = sorted(d.get('models') or ())
    if len(models) != 1:
        raise _refusal('%d models' % len(models), 'it integrates one model')
    if not models[0].endswith('.bngl'):
        raise _refusal('the model %s' % models[0], 'it integrates one BNGL model on the bngsim '
                       '.net path')


def _check_one_experiment(d):
    experiments = [(k[1], v) for k, v in d.items()
                   if isinstance(k, tuple) and k and k[0] == 'experiment']
    if len(experiments) != 1:
        raise _refusal('%d experiment: lines' % len(experiments), 'it assimilates one time series')
    name, fields = experiments[0]
    for label in fields:
        if label in _REFUSED_EXPERIMENT_FIELDS:
            raise PybnfError("job_type = lwf does not read the field '%s:' on the experiment %s."
                             % (label, name), hint='Remove it.')
    kind = str(fields.get('type', 'time_course')).lower()
    if kind != 'time_course':
        raise _refusal('type: %s on the experiment %s' % (kind, name), 'it assimilates a time course')
    method = str(fields.get('method', 'ode')).lower()
    if method != 'ode':
        raise _refusal('method: %s on the experiment %s' % (method, name),
                       'a segment is an ODE integration')
    files = list(fields.get('data') or ())
    constraints = [f for f in files if f.lower().endswith(('.con', '.prop'))]
    if constraints:
        raise _refusal('the constraint file(s) %s' % ', '.join(constraints),
                       'the weights are the noise model on the data rows alone')
    if len(files) != 1:
        raise _refusal('%d data files on the experiment %s' % (len(files), name),
                       'it assimilates one count per row')


def _check_independent_runs(d):
    runs = d.get('population_size', 1)
    if runs != 1:
        raise _refusal('population_size = %s' % runs, 'the number of independent runs is '
                       'lwf_independent_runs' + (' = %d' % runs if runs > 1 else ''))
    if 'parallel_count' in d and d.get('lwf_independent_runs', 1) == 1:
        raise _refusal('parallel_count with one independent run', 'one run starts no cluster')
    if d.get('parallel_count') is not None and int(d['parallel_count']) < 1:
        raise _refusal('parallel_count = %s' % d['parallel_count'],
                       'a cluster needs at least one worker')


# --------------------------------------------------------------------------------------------
# The kernel, resampling and the random streams (pure numpy, kept local per ADR-0009)
# --------------------------------------------------------------------------------------------

def weight_ess(weights):
    """The weight ESS of normalized weights, ``1 / sum(w**2)``."""
    weights = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(weights ** 2))


def systematic_resample(weights, uniform):
    """Ancestor indices by systematic resampling, the positions offset by ``uniform`` in
    ``[0, 1)``: particle ``i`` receives ``floor(N w_i)`` or ``ceil(N w_i)`` copies."""
    weights = np.asarray(weights, dtype=float)
    n = weights.size
    positions = (uniform + np.arange(n)) / n
    # The cumulative sum of normalized weights can end a few ulps below 1, which would hand a
    # position above it to particle n - 1 whatever its weight: clamp to the last positive weight.
    last = np.flatnonzero(weights > 0.0)[-1]
    return np.minimum(np.searchsorted(np.cumsum(weights), positions), last)


def liu_west_move(x, weights, h, z, var_floor=None):
    """One Liu–West move of the rows of ``x`` (particles by parameters, in the working space),
    ``x'_ij = a_j x_ij + (1 - a_j) m_j + h_j (z L^T)_ij`` with ``a_j = sqrt(1 - h_j**2)``.
    ``var_floor`` floors the covariance's diagonal. Returns ``(moved, cholesky)``; ``cholesky`` is
    False when the covariance did not factor and the noise was independent per parameter."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    h = np.broadcast_to(np.asarray(h, dtype=float), (x.shape[1],))
    a = np.sqrt(1.0 - h ** 2)
    # A weight of 0 times a value that is not finite would be nan.
    weighed = np.where((w > 0.0)[:, None], x, 0.0)
    mean = w @ weighed
    dev = weighed - mean
    cov = (dev * w[:, None]).T @ dev
    diag = np.arange(cov.shape[0])
    if var_floor is not None:
        cov[diag, diag] = np.maximum(cov[diag, diag], var_floor)
    try:
        noise = z @ np.linalg.cholesky(cov).T
        cholesky = True
    except np.linalg.LinAlgError:
        noise = z * np.sqrt(cov[diag, diag])
        cholesky = False
    return a * x + (1.0 - a) * mean + h * noise, cholesky


def fold_into_box(u, lo, hi):
    """Fold each column of ``u`` into ``[lo_j, hi_j]`` by reflection at the walls, as
    :meth:`FreeParameter.set_value <pybnf.pset.FreeParameter.set_value>` folds in sampling space."""
    u = np.array(u, dtype=float, copy=True)
    for j in range(u.shape[1]):
        col, low, high = u[:, j], lo[j], hi[j]
        out = (col < low) | (col > high)
        if not out.any():
            continue
        v = col[out]
        if np.isfinite(low) and np.isfinite(high):
            width = high - low
            travelled = np.mod(v - low, 2.0 * width)
            v = low + np.minimum(travelled, 2.0 * width - travelled)
        elif np.isfinite(low):
            v = np.where(v < low, 2.0 * low - v, v)
        elif np.isfinite(high):
            v = np.where(v > high, 2.0 * high - v, v)
        col[out] = np.clip(v, low, high)
    return u


def to_working(u, lo, hi, logit):
    """The kernel's working space: the logit of the position in the box for a ``logit`` column,
    the sampling space itself for the others."""
    x = np.array(u, dtype=float, copy=True)
    for j in np.flatnonzero(logit):
        f = np.clip((x[:, j] - lo[j]) / (hi[j] - lo[j]), _EDGE, 1.0 - _EDGE)
        x[:, j] = np.log(f / (1.0 - f))
    return x


def from_working(x, lo, hi, logit):
    """The inverse of :func:`to_working`, then a fold into the box."""
    u = np.array(x, dtype=float, copy=True)
    for j in np.flatnonzero(logit):
        u[:, j] = lo[j] + (hi[j] - lo[j]) / (1.0 + np.exp(-x[:, j]))
    return fold_into_box(u, lo, hi)


def stream(seed, run, purpose, t=None):
    """The generator for one purpose of one independent run, keyed on the time ``t`` of its row,
    never on a position."""
    key = (int(run), int(purpose))
    if t is not None:
        key += struct.unpack('<II', struct.pack('<d', float(t) + 0.0))
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def _zero_level(c0, c1):
    """The largest increment from ``c0`` to ``c1`` that is integrator noise."""
    return max(_ZERO_ABS, _ZERO_REL * max(abs(c0), abs(c1)))


def _weighted_quantiles(values, weights, qs):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    cum /= cum[-1]
    return [float(values[order][min(np.searchsorted(cum, q), len(values) - 1)]) for q in qs]


def _valued(template, value):
    """``template`` holding ``value`` (inside its box). A copy, because ``set_value`` rebuilds the
    prior, which over every particle at every row costs more than the integration."""
    fp = copy.copy(template)
    fp.value = float(value)
    return fp


# --------------------------------------------------------------------------------------------
# The population and its record
# --------------------------------------------------------------------------------------------

@dataclass
class _Population:
    """One run's particles: parameters in sampling space, carried states, weights, and the
    expected count each ancestral line predicted at every assimilated row."""
    u: np.ndarray
    state: np.ndarray
    weights: np.ndarray
    predicted: list = field(default_factory=list)

    def resample(self, index):
        self.u = self.u[index]
        self.state = self.state[index]
        self.weights = np.full(len(index), 1.0 / len(index))
        self.predicted = [row[index] for row in self.predicted]


@dataclass
class _Update:
    """What one update did, for the weight-ESS history and the summary."""
    time: float
    count: float
    weight_ess: float
    distinct: int
    resampled: bool
    failed: int
    zero_increment: int
    log_evidence: float


class _DegenerateUpdate(Exception):
    """No particle can explain a row."""


@dataclass
class _Sample:
    """One run's equal-weight resample after its last assimilated row (parameters in own units,
    predicted counts) and its forecast counts, None when no forecast was issued."""
    theta: np.ndarray
    predicted: np.ndarray
    forecast_times: tuple = ()
    forecasts: Optional[np.ndarray] = None


@dataclass
class _RunResult:
    """What one independent run hands back for its outputs; it holds no integrator, so it pickles."""
    run: int
    assimilated: int
    history: list
    by_row: list
    sample: _Sample
    start_draws: int
    integrations: int
    budget_stop: bool = False
    degenerate: Optional[str] = None
    messages: Optional[list] = None


def _filter_independent_run(blob, run, budget, saved=None):
    """Independent run ``run`` on a dask worker, on its own unpickled copy of the filter.
    ``budget`` is ``(limit, elapsed, sent)`` with ``sent`` a wall-clock time, since a monotonic
    clock does not carry from one process to another."""
    alg = pickle.loads(blob)
    if budget is not None:
        limit, elapsed, sent = budget
        alg.budget = FitBudget(limit, elapsed=elapsed + max(0.0, time.time() - sent))
    alg._messages = []
    result = alg._filter_run(run, saved)
    result.messages = alg._messages
    return result


# --------------------------------------------------------------------------------------------
# The job type
# --------------------------------------------------------------------------------------------

@register_fit_type('lwf', family=FAMILY, display_name='Liu–West Filter', schema=LWFConfig)
class LiuWestFilter(Algorithm):
    """``job_type = lwf``: one model filtered against one time course."""

    parallelism_setting = None
    # One independent run integrates in this process; several are tasks on a cluster.
    needs_cluster = False

    #: main() refuses -r with this: the filter continues through its state file.
    resume_refusal = (
        '-r / --resume does not apply to job_type = lwf: the Liu–West filter keeps no pickled '
        'algorithm to resume, and continues through its state file instead.',
        'Set lwf_continue = 1 to continue from the state file (lwf_state_file, by default '
        'lwf_state.npz in output_dir), which assimilates only the rows after those it holds.')

    def __init__(self, config):
        _refuse_cluster_flags(config.config)
        self.observable, self.suffix, self._exp = _resolve_observation(config)
        self.times, self.step = _resolve_grid(self._exp, self.suffix)
        _resolve_counts(self._exp, self.observable, self.suffix)
        self._noise_source = _resolve_noise_model(config.obj, self.observable, config.config)
        fixed = isinstance(self._noise_source, ConstantSigma)
        self._dispersion_name = None if fixed else self._noise_source.name
        self._dispersion_value = self._noise_source.const if fixed else None
        _check_parameters(config)
        cf = config.config
        self.particles = int(cf['lwf_particles'])
        self.threshold = float(cf['lwf_resample_threshold'])
        self.forecast_intervals = int(cf['lwf_forecast_intervals'])
        self.bounds_mode = cf['lwf_bounds']
        self.runs = int(cf['lwf_independent_runs'])
        self.needs_cluster = self.runs > 1
        if self.needs_cluster:
            given = cf.get('parallel_count')
            if given is None:
                from ...cluster import Cluster
                workers = min(self.runs, Cluster.cpus_per_node()[0])
            else:
                workers = min(int(given), self.runs)
            if workers != given:
                logger.info('Liu–West filter: %d independent runs on %d worker(s); parallel_count '
                            'was %s', self.runs, workers, given)
                cf['parallel_count'] = workers
        self.continuing = bool(cf['lwf_continue'])
        self.state_files = _state_paths(cf.get('lwf_state_file'), cf['output_dir'],
                                        cf.get('simulation_dir'), self.runs)
        if not self.continuing:
            _refuse_replacing_other_files(self.state_files, cf.get('lwf_state_file'))
        self._jitter_given = cf.get('lwf_jitter')
        self._jitter_lines = _jitter_lines(cf.get('lwf_parameter_jitter') or (), config.variables)
        # The shared Algorithm setup reads both (pybnf.config._NO_SEARCH_RUNS).
        cf.setdefault('population_size', 1)
        cf.setdefault('max_iterations', 1)
        super().__init__(config)
        self.model = self.model_list[0]
        names = tuple(v.name for v in self.variables)
        self._build_integrator(names)
        self._resolve_kernel(names)
        self._rows = [_one_row(self._exp, k) for k in range(len(self.times))]
        self._seed = self._seed_sequence.entropy
        self._seed_given = config.random_seed_given
        self._run = 0
        self._history = []
        self._by_row = []
        self._notes = set()
        # Warnings a run on a worker keeps for run() to print; None prints them at once.
        self._messages = None
        self._fingerprint = self._build_fingerprint()

    # -- construction ----------------------------------------------------------------------

    def _build_integrator(self, names):
        """The integrator over the free parameters that are model parameters. The dispersion is
        written into the model too only when it is one, as the fit path writes every free
        parameter it can."""
        noise = self._dispersion_name
        engine = getattr(self.model, '_engine_model', None)   # None: the integrator refuses it
        model_ids = [n for n in names if n != noise or n in getattr(engine, 'param_names', ())]
        integrator = SegmentIntegrator(self.model, model_ids, suffix=self.suffix,
                                       timeout=self.config.config.get('wall_time_sim'))
        outputs = set(engine.observable_names) | set(engine.function_names)
        if self.observable not in outputs:
            raise PybnfError(
                "The data column '%s' of the experiment %s is not an observable or function of "
                "the model %s, so the filter has nothing to score it against. Its outputs are %s."
                % (self.observable, self.suffix, self.model.name, ', '.join(sorted(outputs))))
        self.integrator = integrator
        self._model_columns = np.array([names.index(n) for n in integrator.parameter_names],
                                       dtype=int)
        self._dispersion_column = None if noise is None else names.index(noise)

    def _resolve_kernel(self, names):
        """Each parameter's h, the parameters the kernel never moves, and the box it moves in."""
        start_only = initial_state_only_ids(self.model.netfile_lines,
                                            [n for n in names if n != self._dispersion_name])
        named = [n for n in self._jitter_lines if n in start_only]
        if named:
            raise PybnfError(
                'lwf_parameter_jitter names %s, which acts on the model only through its initial '
                'state, so the kernel never moves it and the jitter would be ignored.' % named[0],
                hint='Remove the lwf_parameter_jitter line for %s.' % named[0])
        moved = [n for n in names if n not in start_only]
        if self._jitter_given is not None and moved and all(n in self._jitter_lines for n in moved):
            raise PybnfError(
                'lwf_jitter = %r is set, but every parameter the kernel moves has an '
                'lwf_parameter_jitter line (%s), so lwf_jitter would be ignored.'
                % (self._jitter_given, ', '.join(moved)),
                hint='Remove the lwf_jitter line, or the lwf_parameter_jitter line of a parameter '
                     'that should use it.')
        default = DEFAULT_JITTER if self._jitter_given is None else float(self._jitter_given)
        self.start_only = tuple(start_only)
        self.jitter = np.array([0.0 if n in start_only else self._jitter_lines.get(n, default)
                                for n in names])
        self._moved = self.jitter > 0.0
        lo, hi = zip(*(_box_in_sampling_space(v) for v in self.variables))
        self._lo, self._hi = np.array(lo), np.array(hi)
        boxed = np.isfinite(self._lo) & np.isfinite(self._hi) & (self._hi > self._lo)
        self._logit = boxed if self.bounds_mode == 'logit' else np.zeros(len(names), dtype=bool)
        self._theta_lo = np.array([v.lower_bound for v in self.variables], dtype=float)
        self._theta_hi = np.array([v.upper_bound for v in self.variables], dtype=float)

    def _build_fingerprint(self):
        """Everything that shapes the arithmetic up to the last row, as text a refusal can quote.
        ``lwf_forecast_intervals`` is left out, so a continuation may change it."""
        names = [v.name for v in self.variables]
        return {
            'pybnf version': __version__,
            'bngsim version': str(_bngsim_caps.BNGSIM_VERSION),
            'bngsim build': _bngsim_caps.bngsim_build_id() or 'unknown',
            'model network': _network_digest(self.model.netfile_lines, names),
            'species': ', '.join(self.integrator.species_names),
            'free parameters': ', '.join(names),
            'priors and bounds': '; '.join(_prior_record(v) for v in self.variables),
            'observable': self.observable,
            'noise model': COLUMN_FORM.format(column=self.observable,
                                              dispersion=_dispersion_text(self._noise_source)),
            'start-only parameters': ', '.join(self.start_only) or 'none',
            'jitter': ', '.join('%s %r' % (n, float(h)) for n, h in zip(names, self.jitter)),
            'lwf_bounds': self.bounds_mode,
            'lwf_particles': str(self.particles),
            'lwf_resample_threshold': repr(self.threshold),
            'random_seed': str(self._seed),
            'initialization': str(self.config.config['initialization']),
            'lwf_independent_runs': str(self.runs),
        }

    # -- the run-loop contract (not used: the filter drives its own run) ----------------------

    def start_run(self):
        raise PybnfError('job_type = lwf drives its own run; this is an internal wiring error.')

    def got_result(self, res):
        raise PybnfError('job_type = lwf drives its own run; this is an internal wiring error.')

    def cleanup(self):
        """The filter keeps no trajectory of best fits for the base class to write."""

    # -- the run ---------------------------------------------------------------------------

    def run(self, client=None, resume=None, debug=False):
        """Filter every data row, then forecast, for each independent run: in this process, or
        one task each on ``client`` when there are several."""
        self.stop_reason = None
        self.completed_simulations = 0
        print2(self._banner())
        saved = self._load_states() if self.continuing else [None] * self.runs
        results = self._filter_runs(client, saved)
        self.completed_simulations = sum(r.integrations for r in results)
        stopped = [r for r in results if r.budget_stop]
        if stopped:
            self._stop_for_budget(results)
        self._write_outputs(results)
        print1('Liu–West filter: ' + SAMPLE_STATEMENT)
        degenerate = [r for r in results if r.degenerate]
        if degenerate:
            if self.runs == 1:
                message = degenerate[0].degenerate
            else:
                message = 'Degenerate update in %d of %d independent runs. %s' % (
                    len(degenerate), self.runs,
                    ' '.join('Run %d: %s' % (r.run, r.degenerate) for r in degenerate))
            raise PybnfError(message, hint=[
                'Check the data row: a count the model cannot produce is often a data error, '
                'which a nan row leaves unscored. After correcting it, lwf_continue = 1 '
                'continues from the state file, unless another independent run already '
                'assimilated that row; then start a fresh run.',
                'Otherwise widen the priors, raise lwf_particles, or raise the jitter.'])
        self._teardown_sim_dir()

    def _filter_runs(self, client, saved):
        """Every independent run's result, in run order. ``saved`` holds each run's state to
        continue from, or None to start from the prior."""
        if client is None or self.runs == 1:
            return [self._filter_run(run, saved[run]) for run in range(self.runs)]
        blob = client.scatter(pickle.dumps(self), broadcast=True)
        budget = (None if self.budget is None
                  else (self.budget.limit, self.budget.elapsed(), time.time()))
        futures = [client.submit(_filter_independent_run, blob, run, budget, saved[run],
                                 pure=False)
                   for run in range(self.runs)]
        results = []
        for future in futures:
            result = future.result()
            self._emit(result.messages)
            results.append(result)
        return results

    def _filter_run(self, run, saved=None):
        """Filter independent run ``run`` from the prior draw, or from ``saved``, through every
        row it can assimilate, saving its state after each, then draw its output sample and,
        unless a degenerate update ended it, its forecast."""
        self._run = run
        self._notes = set()
        before = self.integrator.n_integrations
        if saved is None:
            self._history, self._by_row = [], []
            population = self._initial_population()
            assimilated = 0
            self._save_state(population, assimilated)
        else:
            population, assimilated = self._restore(saved)
        rows = len(self.times)
        stopped, degenerate = False, None
        try:
            for k in range(assimilated, rows):
                if self._budget_spent():
                    stopped = True
                    break
                self._update(population, k)
                assimilated = k + 1
                self._save_state(population, assimilated)
        except _DegenerateUpdate as exc:
            degenerate = str(exc)
        sample = self._sample(population, assimilated, forecast=degenerate is None)
        return _RunResult(run=run, assimilated=assimilated, history=self._history,
                          by_row=self._by_row, sample=sample,
                          start_draws=self._distinct_start_draws(population),
                          integrations=self.integrator.n_integrations - before,
                          budget_stop=stopped, degenerate=degenerate)

    # -- the state file --------------------------------------------------------------------

    def _save_state(self, pop, assimilated):
        """Write the run's state file through row ``assimilated``."""
        names = [v.name for v in self.variables]
        rows = self._rows[:assimilated]
        arrays = {
            'format': np.array(STATE_FORMAT),
            'format_version': np.array(STATE_VERSION),
            'fingerprint': np.array(json.dumps(self._fingerprint, sort_keys=True)),
            'run': np.array(self._run),
            'u': pop.u,
            'state': pop.state,
            'weights': pop.weights,
            'predicted': np.array(pop.predicted, dtype=float).reshape(assimilated,
                                                                     self.particles),
            'times': np.asarray(self.times[:assimilated], dtype=float),
            'counts': np.array([float(r.data[0, r.cols[self.observable]]) for r in rows]),
            'history': np.array([[u.time, u.count, u.weight_ess, u.distinct, u.resampled,
                                  u.failed, u.zero_increment, u.log_evidence]
                                 for u in self._history], dtype=float).reshape(assimilated, 8),
            'by_row': np.array([[t] + [x for row in summary for x in row]
                                for t, summary in self._by_row], dtype=float
                               ).reshape(assimilated, 1 + 4 * len(names)),
            'var_floor': np.asarray(self._var_floor, dtype=float),
        }
        path = self.state_files[self._run]
        try:
            _write_state(path, arrays)
        except OSError as exc:
            raise PybnfError(
                'The Liu–West filter could not write its state file %s after row %d: %s. The '
                'file there before, if any, is unchanged.' % (path, assimilated, exc),
                hint='Point lwf_state_file at a folder that can be written to, then run again.'
            ) from None

    def _restore(self, saved):
        """``(population, assimilated)`` from a state file, with the histories and the floor."""
        k = saved['times'].shape[0]
        names = [v.name for v in self.variables]
        self._history = [_Update(float(h[0]), float(h[1]), float(h[2]), int(h[3]), bool(h[4]),
                                 int(h[5]), int(h[6]), float(h[7])) for h in saved['history']]
        self._by_row = [(float(row[0]), [[float(x) for x in row[1 + 4 * j:5 + 4 * j]]
                                         for j in range(len(names))])
                        for row in saved['by_row']]
        self._var_floor = np.array(saved['var_floor'], dtype=float)
        population = _Population(u=np.array(saved['u'], dtype=float),
                                 state=np.array(saved['state'], dtype=float),
                                 weights=np.array(saved['weights'], dtype=float),
                                 predicted=[np.array(row, dtype=float)
                                            for row in saved['predicted']])
        return population, k

    def _load_states(self):
        """Every run's state file, checked against this configuration and data before any run
        starts."""
        saved = []
        for run, path in enumerate(self.state_files):
            state, held = self._load_state(run, path)
            if run == 0 and not self._seed_given:
                self._seed = int(held['random_seed'])
                self._fingerprint['random_seed'] = str(self._seed)
                self.config.config['random_seed'] = self._seed
                logger.info('Random seed: %d, taken from the state file %s since the configuration '
                            'sets none; the seed drawn for this run is not used.', self._seed, path)
            self._check_fingerprint(path, held)
            self._check_integrity(path, state)
            self._check_rows(path, state)
            saved.append(state)
        rows = len(self.times)
        if all(state['times'].shape[0] == rows for state in saved):
            message = ('Warning: lwf_continue = 1 found no row after the %d the state file%s '
                       'hold%s, so nothing was assimilated, and the outputs are written again '
                       'from %s.' % (rows, '' if self.runs == 1 else 's',
                                     's' if self.runs == 1 else '',
                                     'it' if self.runs == 1 else 'them'))
            logger.warning(message)
            print0(message)
        return saved

    def _load_state(self, run, path):
        """``(arrays, fingerprint)`` of run ``run``'s state file, refused when it is missing, is
        not a state file of this version for this run, or holds entries of another type."""
        if not os.path.isfile(path):
            raise PybnfError('lwf_continue = 1 asks the Liu–West filter to continue from the state '
                             'file %s, which does not exist.' % path,
                             hint=['Point lwf_state_file at the file an earlier run wrote.',
                                   _FRESH_RUN])

        def refused(problem):
            return PybnfError('lwf_continue = 1 cannot continue from the state file %s: %s.'
                              % (path, problem), hint=_FRESH_RUN)

        try:
            saved = _read_state(path)
        except Exception as exc:
            raise refused('it cannot be read (%s)' % exc) from None
        if str(saved.get('format', '')) != STATE_FORMAT:
            raise refused('it is not a state file of the Liu–West filter')
        version = saved.get('format_version')
        if (version is None or version.shape != () or not np.issubdtype(version.dtype, np.integer)
                or int(version) != STATE_VERSION):
            raise refused('it is not of state-file format version %d' % STATE_VERSION)
        missing = [key for key in STATE_KEYS if key not in saved]
        if missing:
            raise refused('it lacks %s' % ', '.join(missing))
        held_run = saved['run']
        if (held_run.shape != () or not np.issubdtype(held_run.dtype, np.integer)
                or int(held_run) != run):
            raise refused('it does not hold independent run %d' % run)
        held = None
        if saved['fingerprint'].shape == () and saved['fingerprint'].dtype.kind == 'U':
            with contextlib.suppress(ValueError):
                held = json.loads(str(saved['fingerprint']))
        if not (isinstance(held, dict) and all(isinstance(v, str) for v in held.values())
                and held.get('random_seed', '').isdecimal()):
            raise refused('its fingerprint is not a JSON object of strings holding a random_seed')
        floats = [key for key in ('u', 'state', 'weights', 'predicted', 'times', 'counts',
                                  'history', 'by_row', 'var_floor')
                  if saved[key].dtype != np.float64]
        if floats:
            raise refused('%s %s not float64' % (', '.join(floats),
                                                 'is' if len(floats) == 1 else 'are'))
        return saved, held

    def _check_fingerprint(self, path, held):
        """Refuse a state file written under another configuration, naming what differs."""
        now = self._fingerprint
        differs = ['%s: %s -> %s' % (key, held.get(key, '(none)'), now.get(key, '(none)'))
                   for key in list(now) + sorted(set(held) - set(now))
                   if held.get(key) != now.get(key)]
        if differs:
            raise PybnfError(
                'lwf_continue = 1 cannot continue from the state file %s: it was written under a '
                'configuration that differs from this one. What differs (the state file -> this '
                'configuration): %s.' % (path, '; '.join(differs)),
                hint=['Restore what differs, as it was when the state file was written.',
                      _FRESH_RUN])

    def _check_integrity(self, path, saved):
        """Refuse arrays of another shape than this configuration gives them, or holding a value
        the filter never writes (a predicted count or a row's count may be nan, and a particle
        without weight may hold a prior draw that is not finite)."""
        k = saved['times'].shape[0] if saved['times'].ndim == 1 else -1
        n, d = self.particles, len(self.variables)
        want = {'times': (k,), 'u': (n, d), 'state': (n, len(self.integrator.species_names)),
                'weights': (n,), 'predicted': (k, n), 'counts': (k,), 'history': (k, 8),
                'by_row': (k, 1 + 4 * d), 'var_floor': (int(self._moved.sum()),)}
        bad = [key for key, shape in want.items() if saved[key].shape != shape]
        problem = 'not of the shape this configuration gives'
        if not bad:
            finite = {key: saved[key] for key in ('u', 'state', 'weights', 'var_floor', 'times',
                                                  'by_row')}
            finite['u'] = saved['u'][saved['weights'] > 0.0]
            finite['history'] = np.delete(saved['history'], 1, axis=1)   # but the count
            bad = [key for key, values in finite.items() if not np.isfinite(values).all()]
            problem = 'holding a value that is not a finite number'
        if bad:
            raise PybnfError('The state file %s is damaged (%s %s), so lwf_continue = 1 cannot '
                             'continue from it.' % (path, ', '.join(bad), problem), hint=_FRESH_RUN)

    def _check_rows(self, path, saved):
        """Refuse data that no longer hold the rows the state file assimilated: the filter cannot
        take back a row it has weighed."""
        times, counts = saved['times'], saved['counts']
        k = times.shape[0]
        if len(self.times) < k:
            raise PybnfError(
                'The data of the experiment %s hold %d row(s), fewer than the %d the state file '
                '%s assimilated, so lwf_continue = 1 cannot continue from it.'
                % (self.suffix, len(self.times), k, path),
                hint=['Restore the rows, if they were removed by mistake.', _FRESH_RUN])
        for j, row in enumerate(self._rows[:k]):
            now = float(row.data[0, row.cols[self.observable]])
            if times[j] != self.times[j]:
                what = 'its time was %r and is %r now' % (float(times[j]), float(self.times[j]))
            elif not (counts[j] == now or (math.isnan(counts[j]) and math.isnan(now))):
                what = 'its count was %r and is %r now' % (float(counts[j]), now)
            else:
                continue
            raise PybnfError('lwf_continue = 1 cannot continue from the state file %s: row %d of '
                             'the data changed since it was assimilated (%s), and the filter '
                             'cannot take a row back.' % (path, j + 1, what), hint=_FRESH_RUN)

    def __setstate__(self, state):
        """A copy on a worker; Algorithm's would look for a trajectory backup to resume from."""
        self.__dict__.update(state)

    def _banner(self):
        runs = ('' if self.runs == 1 else
                '%d independent runs of ' % self.runs)
        return ('Running the Liu–West filter: %s%d particles over %d rows of %s (step %g), '
                'forecasting %d step(s) past the last row'
                % (runs, self.particles, len(self.times), self.suffix, self.step,
                   self.forecast_intervals))

    def _reached(self, k):
        if k == 0:
            return 'before assimilating any row'
        return 'after assimilating row %d of %d (t = %g)' % (k, len(self.times), self.times[k - 1])

    def _stop_for_budget(self, results):
        """Say once, in the log and ``stop_reason.txt``, which rows each run assimilated before
        the budget ran out; each forecast starts at that row (ADR-0093)."""
        reached = [r.assimilated for r in results]
        if self.runs == 1:
            where = 'stopped %s' % self._reached(reached[0])
            outputs = ('The parameters written are the prior draw, the per-row histories are '
                       'empty' if reached[0] == 0 else 'The filtered outputs go through that row')
        elif len(set(reached)) == 1:
            where = 'stopped %s in each of its %d independent runs' % (self._reached(reached[0]),
                                                                     self.runs)
            outputs = ('The parameters written, per run and combined, are the prior draws, the '
                       'per-row histories are empty' if reached[0] == 0 else
                       'The filtered outputs, per run and combined, go through that row')
        else:
            where = 'stopped its %d independent runs at different rows: %s' % (
                self.runs, '; '.join('run %d %s' % (r.run, self._reached(r.assimilated))
                                     for r in results))
            outputs = ("The combined parameters and predicted counts are not written, since the "
                       "runs reached no common row, each run's own outputs (%s/run_<r>/) go "
                       "through its last row" % OUTPUT_FOLDER)
        self.stop_reason = (
            'Wall-time budget reached: the Liu–West filter %s, %s into the run '
            '(wall_time_fit = %g s) and after %d segment integration(s). %s, and each forecast '
            'starts at the last row assimilated, not at the end of the data.'
            % (where, format_duration(self.budget.elapsed()), self.budget.limit,
               self.completed_simulations, outputs))
        self._announce_stop_reason()
        held = ('the state file (%s) holds the population through the last row assimilated'
                % self.state_files[0] if self.runs == 1 else
                'the state files (%s, ...) hold each run through the last row it assimilated'
                % self.state_files[0])
        print1('  -> Continue with lwf_continue = 1: %s, and a continuation writes what an '
               'uninterrupted run writes.' % held)
        print1('  -> Or raise wall_time_fit, or set it to 0 for no limit, to assimilate every row '
               'and forecast in one run.')

    # -- the initial population ------------------------------------------------------------

    def _initial_population(self):
        """A prior draw in sampling space, a Latin hypercube over the bounded parameters under
        ``initialization = lh``, each particle at the initial state its parameters set."""
        n, variables = self.particles, self.variables
        rng = stream(self._seed, self._run, _PRIOR_DRAW)
        quantiles = np.empty((n, len(variables)))
        bounded = [v.has_bounded_initialization for v in variables]
        strata = (latin_hypercube(n, sum(bounded), rng)
                  if self.config.config['initialization'] == 'lh' else None)
        column = 0
        for j, v in enumerate(variables):
            if strata is not None and bounded[j]:
                quantiles[:, j] = strata[:, column]
                column += 1
            else:
                quantiles[:, j] = rng.random(n)
        # Away from 0 and 1, where an open prior's inverse CDF is infinite.
        quantiles = np.clip(quantiles, 2.0 ** -53, 1.0 - 2.0 ** -53)
        u = np.array([[v.prior_quantile_u(q) for q in quantiles[:, j]]
                      for j, v in enumerate(variables)]).T
        u = np.clip(u, self._lo, self._hi)
        theta = self._theta(u)
        state = np.zeros((n, len(self.integrator.species_names)))
        weights = np.full(n, 1.0 / n)
        for i in range(n):
            if not np.all(np.isfinite(theta[i])):
                weights[i] = 0.0
                continue
            try:
                state[i] = self.integrator.initial_state(theta[i, self._model_columns])
            except SegmentFailed:
                weights[i] = 0.0
        if not weights.any():
            raise PybnfError('No particle of the prior draw has a finite initial state, so there '
                             'is nothing to filter.', hint='Check the model and the priors.')
        weights /= weights.sum()
        moved, live = self._moved, weights > 0.0
        self._var_floor = _VAR_FLOOR_FRAC * np.var(
            to_working(u[live][:, moved], self._lo[moved], self._hi[moved], self._logit[moved]),
            axis=0)
        return _Population(u=u, state=state, weights=weights)

    def _theta(self, u):
        """Parameter values from sampling space, clipped into each parameter's box."""
        theta = np.empty_like(u)
        for j, v in enumerate(self.variables):
            theta[:, j] = v.from_sampling_space(u[:, j])
        return np.clip(theta, self._theta_lo, self._theta_hi)

    # -- one update ------------------------------------------------------------------------

    def _update(self, pop, k):
        """Assimilate row ``k``: move, integrate, weigh, and resample if the weights degenerate.
        A degenerate update leaves ``pop`` as the last assimilated row left it."""
        t0 = 0.0 if k == 0 else float(self.times[k - 1])
        t1 = float(self.times[k])
        rng = stream(self._seed, self._run, _UPDATE, t1)
        trial = _Population(u=pop.u.copy(), state=pop.state.copy(), weights=pop.weights)
        self._move(trial, rng)
        theta = self._theta(trial.u)
        count = float(self._rows[k].data[0, self._rows[k].cols[self.observable]])
        loglik, predicted, failed, zero = self._weigh(trial, theta, t0, t1, k, count)
        live = pop.weights > 0.0
        with np.errstate(divide='ignore'):
            logw = np.where(live, np.log(np.where(live, pop.weights, 1.0)) + loglik, -np.inf)
        if not np.isfinite(logw).any():
            raise _DegenerateUpdate(
                'Degenerate update at row %d (t = %g, count %g): no particle can explain it. Of '
                'the %d particle(s) with weight, %d failed to integrate and %d predicted no '
                'increase of %s over the interval, which cannot produce a positive count.'
                % (k + 1, t1, count, int(live.sum()), failed, zero, self.observable))
        if math.isnan(count) and failed == 0:
            weights, log_evidence = pop.weights, 0.0
        else:
            top = float(np.max(logw))
            w = np.exp(logw - top)
            log_evidence = top + math.log(float(w.sum()))
            weights = w / w.sum()
        pop.u, pop.state, pop.weights = trial.u, trial.state, weights
        pop.predicted.append(predicted)
        ess = weight_ess(pop.weights)
        resampled = ess < self.threshold * len(pop.weights)
        if resampled:
            pop.resample(systematic_resample(pop.weights, rng.random()))
        distinct = int(np.unique(pop.u[pop.weights > 0.0], axis=0).shape[0])
        self._history.append(_Update(t1, count, ess, distinct, resampled, failed, zero,
                                     log_evidence))
        self._by_row.append((t1, self._summaries(self._theta(pop.u), pop.weights)))
        if distinct < max(2.0, _DISTINCT_WARN_FRAC * len(pop.weights)):
            self._note_once('impoverished',
                            'only %d distinct particle(s) of %d after the update at t = %g: the '
                            'population is impoverished, and its intervals rest on a handful of '
                            'points.' % (distinct, len(pop.weights), t1))

    def _move(self, pop, rng):
        """The Liu–West move of the parameters the kernel moves; the others keep their values."""
        moved = self._moved
        if not moved.any():
            return
        lo, hi, logit = self._lo[moved], self._hi[moved], self._logit[moved]
        x = to_working(pop.u[:, moved], lo, hi, logit)
        z = rng.standard_normal(x.shape)
        new, cholesky = liu_west_move(x, pop.weights, self.jitter[moved], z, self._var_floor)
        if not cholesky:
            self._note_once('cholesky',
                            'the weighted covariance of the moved parameters was not positive '
                            'definite at least once; those moves used independent noise at each '
                            'parameter\'s own scale.')
        pop.u[:, moved] = from_working(new, lo, hi, logit)

    def _weigh(self, pop, theta, t0, t1, k, count):
        """Integrate every live particle over ``[t0, t1]`` and score it on row ``k``: returns
        ``(loglik, predicted increment, failed, zero-increment)``."""
        n = len(pop.weights)
        loglik = np.full(n, -np.inf)
        predicted = np.full(n, np.nan)
        failed = zero = 0
        column = self.observable
        for i in np.flatnonzero(pop.weights > 0.0):
            values = theta[i, self._model_columns]
            if not np.all(np.isfinite(theta[i])):
                failed += 1
                continue
            try:
                segment = self.integrator.integrate(pop.state[i], values, t0, t1)
            except SegmentFailed:
                failed += 1
                continue
            pop.state[i] = segment.state
            rows = segment.data.data
            c0, c1 = float(rows[0, segment.data.cols[column]]), float(rows[-1, segment.data.cols[column]])
            predicted[i] = c1 - c0
            level = _zero_level(c0, c1)
            if c1 - c0 < -level:
                self._note_once('backwards',
                                '%s decreased over an interval under some particles (first at '
                                't = %g), so it is not cumulative under those parameters.'
                                % (column, t1))
            if count > 0 and c1 - c0 <= level:
                zero += 1
                continue
            score = self.objective.evaluate_multiple(
                {self.model.name: {self.suffix: segment.data}},
                {self.model.name: {self.suffix: self._rows[k]}},
                self._pset(theta[i]), show_warnings=(k == 0))
            if score is None or not np.isfinite(score):
                failed += 1
                continue
            loglik[i] = -score
        return loglik, predicted, failed, zero

    def _pset(self, theta_row):
        return PSet([_valued(v, theta_row[j]) for j, v in enumerate(self.variables)])

    def _summaries(self, theta, weights):
        """Each parameter's weighted mean and 2.5, 50 and 97.5 percent quantiles. A particle
        without weight is left out of the mean, since 0 times a value that is not finite is nan."""
        live = weights > 0.0
        rows = []
        for j in range(theta.shape[1]):
            rows.append([float(weights @ np.where(live, theta[:, j], 0.0)),
                         *_weighted_quantiles(theta[:, j], weights, (0.025, 0.5, 0.975))])
        return rows

    def _note_once(self, key, message):
        """A warning, once per independent run: printed now, or kept on a worker for run()."""
        if key in self._notes:
            return
        self._notes.add(key)
        text = 'Warning: %s%s' % ('' if self.runs == 1 else 'independent run %d: ' % self._run,
                                  message)
        if self._messages is None:
            self._emit([text])
        else:
            self._messages.append(text)

    @staticmethod
    def _emit(messages):
        for text in messages or ():
            logger.warning(text)
            print0(text)

    # -- outputs ---------------------------------------------------------------------------

    def _sample(self, pop, assimilated, forecast):
        """An equal-weight resample of the population after its last assimilated row and, when
        ``forecast`` is set, the forecast from it, both on the forecast stream of that row."""
        origin = float(self.times[assimilated - 1]) if assimilated else 0.0
        rng = stream(self._seed, self._run, _FORECAST, origin)
        index = systematic_resample(pop.weights, rng.random())
        theta = self._theta(pop.u[index])
        predicted = (np.column_stack([row[index] for row in pop.predicted]) if pop.predicted
                     else np.zeros((len(index), 0)))
        if not forecast or not self.forecast_intervals:
            return _Sample(theta=theta, predicted=predicted)
        times, draws = self._forecast(pop.state[index].copy(), theta, origin, rng)
        return _Sample(theta=theta, predicted=predicted, forecast_times=tuple(times),
                       forecasts=draws)

    def _write_outputs(self, results):
        """Write each run's files and summary (under ``run_<r>/`` for several) and, for several,
        the combined ones."""
        folder = Path(self.res_dir) / OUTPUT_FOLDER
        if self.runs == 1:
            self._write_run(folder, results[0], [])
            lines = self._run_summary(results[0])
        else:
            for result in results:
                own = folder / ('run_%d' % result.run)
                self._write_run(own, result, ['Independent run %d of %d.' % (result.run,
                                                                           self.runs)])
                self._write_summary(own, self._run_summary(result), console=False)
            lines = self._combined_summary(results, self._write_combined(folder, results))
        self._write_summary(folder, lines)

    def _write_run(self, folder, result, own):
        """One run's files in ``folder``, each opened by the lines ``own``."""
        folder.mkdir(parents=True, exist_ok=True)
        names = [v.name for v in self.variables]
        sample, history, k = result.sample, result.history, result.assimilated
        _write_table(folder / 'parameters.txt', own + [self._parameters_note(k)], names,
                     sample.theta)
        _write_table(folder / 'parameters_by_row.txt',
                     own + ['Each parameter\'s weighted mean and 2.5, 50 and 97.5 percent '
                            'quantiles after the update at each row.'],
                     ['time'] + ['%s_%s' % (n, s) for n in names
                                 for s in ('mean', 'q2.5', 'q50', 'q97.5')],
                     np.array([[t] + [x for row in rows for x in row]
                               for t, rows in result.by_row]
                              ).reshape(len(result.by_row), 1 + 4 * len(names)))
        _write_table(folder / 'predicted_counts.txt', own + [self._predicted_note()],
                     _time_headers(self.times[:k]), sample.predicted)
        _write_table(folder / 'weight_ess.txt',
                     own + ['One row per assimilated row: its time and observed count, the '
                            'weight ESS after weighing (before any resample), the number of '
                            'particles, the distinct particles after the update, whether it '
                            'resampled, the failed and zero-increment particles, and the row\'s '
                            'log evidence under the drifting-parameter model.'],
                     ['time', 'count', 'weight_ess', 'particles', 'distinct', 'resampled',
                      'failed', 'zero_increment', 'log_evidence'],
                     np.array([[u.time, u.count, u.weight_ess, self.particles, u.distinct,
                                int(u.resampled), u.failed, u.zero_increment, u.log_evidence]
                               for u in history]).reshape(len(history), 9))
        if sample.forecasts is not None:
            _write_table(folder / 'forecasts.txt', own + [self._forecasts_note()],
                         _time_headers(sample.forecast_times), sample.forecasts)

    def _write_combined(self, folder, results):
        """Stack each run's own resample in run order, only when every run reached the same row;
        returns whether they were written."""
        reached = {r.assimilated for r in results}
        if len(reached) != 1:
            return False
        k, n = reached.pop(), self.particles
        names = [v.name for v in self.variables]

        def how(name):
            return ('The %d independent runs combined with equal weight: rows %d r + 1 to %d (r + '
                    '1) are independent run r\'s own equal-weight resample (run_<r>/%s), whatever '
                    'its log evidence.' % (self.runs, n, n, name))

        _write_table(folder / 'parameters.txt', [how('parameters.txt'), self._parameters_note(k)],
                     names, np.vstack([r.sample.theta for r in results]))
        _write_table(folder / 'predicted_counts.txt',
                     [how('predicted_counts.txt'), self._predicted_note()],
                     _time_headers(self.times[:k]),
                     np.vstack([r.sample.predicted for r in results]))
        if all(r.sample.forecasts is not None for r in results):
            _write_table(folder / 'forecasts.txt', [how('forecasts.txt'), self._forecasts_note()],
                         _time_headers(results[0].sample.forecast_times),
                         np.vstack([r.sample.forecasts for r in results]))
        return True

    def _parameters_note(self, k):
        through = ('through row %d of %d (t = %g)' % (k, len(self.times), self.times[k - 1])
                   if k else 'before any row was assimilated')
        return ('One row per particle of an equal-weight resample of the population %s, in each '
                'parameter\'s own units.' % through)

    def _predicted_note(self):
        return ('The expected count at each assimilated row (the increase of %s over the interval '
                'ending at that time) along the ancestral line of each particle of the same '
                'resample.' % self.observable)

    def _forecasts_note(self):
        return ('Counts drawn from the job\'s noise model (negative binomial, the prediction taken '
                'as its mean, each particle\'s own dispersion) around the increase of %s over each '
                'forecast interval, one row per particle of the same resample; nan where a '
                'particle failed to integrate.' % self.observable)

    def _forecast(self, state, theta, origin, rng):
        """Integrate each particle ``lwf_forecast_intervals`` steps on and draw a count for each."""
        n = theta.shape[0]
        times = [origin + self.step * (j + 1) for j in range(self.forecast_intervals)]
        draws = np.full((n, len(times)), np.nan)
        alive = np.ones(n, dtype=bool)
        dispersion = (np.full(n, self._dispersion_value) if self._dispersion_column is None
                      else theta[:, self._dispersion_column])
        t0 = origin
        for j, t1 in enumerate(times):
            mean = np.full(n, np.nan)
            for i in np.flatnonzero(alive):
                try:
                    segment = self.integrator.integrate(state[i], theta[i, self._model_columns],
                                                        t0, t1)
                except SegmentFailed:
                    alive[i] = False
                    continue
                state[i] = segment.state
                col = segment.data.cols[self.observable]
                c0, c1 = float(segment.data.data[0, col]), float(segment.data.data[-1, col])
                if not (np.isfinite(c0) and np.isfinite(c1)):
                    alive[i] = False
                    continue
                mean[i] = c1 - c0 if c1 - c0 > _zero_level(c0, c1) else 0.0
            if not alive.any():
                raise PybnfError('Every particle failed to integrate by the forecast interval '
                                 'ending at t = %g, so there is no forecast to write.' % t1)
            r = dispersion[alive]
            draws[alive, j] = rng.negative_binomial(r, r / (r + mean[alive]))
            t0 = t1
        return times, draws

    def _moved_text(self):
        names = [v.name for v in self.variables]
        return ', '.join('%s at h = %r' % (n, float(h)) for n, h in zip(names, self.jitter)
                         if h > 0) or 'none'

    def _rows_text(self, k):
        return 'Rows assimilated: %d of %d%s.' % (
            k, len(self.times), '' if not k else ' (t = %g to %g)' % (self.times[0],
                                                                     self.times[k - 1]))

    def _run_stop_text(self, result):
        """Why a run ended early, or None."""
        if result.degenerate:
            return result.degenerate
        if not result.budget_stop:
            return None
        if self.runs == 1:
            return self.stop_reason
        return ('Wall-time budget reached: this run stopped %s; Results/stop_reason.txt gives '
                'the reason for every run.' % self._reached(result.assimilated))

    def _run_summary(self, result):
        """One run's summary as ``(lines for summary.txt, lines for the console)``."""
        if self.runs == 1:
            head = 'one independent run'
        else:
            head = 'independent run %d of %d' % (result.run, self.runs)
        lines = [
            'Liu–West filter (job_type = lwf), %s, %d particles, random_seed %d.'
            % (head, self.particles, self._seed),
            SAMPLE_STATEMENT,
            self._rows_text(result.assimilated),
            'The kernel moves %s.' % self._moved_text(),
        ]
        if self.start_only:
            lines.append('Never moved, since they enter the model only through its initial state: '
                         '%s. %d distinct initial draw(s) of them remain among the particles with '
                         'weight.' % (', '.join(self.start_only), result.start_draws))
        history = result.history
        if history:
            ess = [u.weight_ess for u in history]
            lines.append('Weight ESS: %.1f after the last row, minimum %.1f; resampled at %d of %d '
                         'rows.' % (ess[-1], min(ess), sum(u.resampled for u in history),
                                    len(history)))
            lines.append('Particles failed: %d; zero-increment particles: %d (summed over rows).'
                         % (sum(u.failed for u in history),
                            sum(u.zero_increment for u in history)))
            lines.append('Log evidence of the drifting-parameter model: %.4f.'
                         % sum(u.log_evidence for u in history))
        stopped = self._run_stop_text(result)
        times = result.sample.forecast_times
        if times:
            lines.append('Forecast: %d interval(s), to t = %g.' % (len(times), times[-1]))
        elif self.forecast_intervals and stopped:
            lines.append('No forecast was written.')
        printed = lines[2:]
        if stopped:
            lines.append('Stopped early: %s' % stopped)
        return lines, printed

    def _combined_summary(self, results, combined):
        """The summary of several runs, as ``(lines for summary.txt, lines for the console)``."""
        lines = [
            'Liu–West filter (job_type = lwf), %d independent runs of %d particles each, '
            'random_seed %d.' % (self.runs, self.particles, self._seed),
            SAMPLE_STATEMENT,
            'The independent runs are combined with equal weight: the combined parameters, '
            'predicted counts and forecasts stack each run\'s own equal-weight resample of its %d '
            'particles, in run order, whatever its log evidence. Each run\'s own files, its '
            'per-row histories among them, are in run_<r>/.' % self.particles,
        ]
        for r in results:
            k = r.assimilated
            line = 'Run %d: %d of %d rows assimilated%s' % (
                r.run, k, len(self.times),
                '' if not k else ' (t = %g to %g)' % (self.times[0], self.times[k - 1]))
            if r.history:
                ess = [u.weight_ess for u in r.history]
                line += ('; weight ESS %.1f after the last row, minimum %.1f; log evidence %.4f'
                         % (ess[-1], min(ess), sum(u.log_evidence for u in r.history)))
            lines.append(line + '.')
        lines.append('The kernel moves %s.' % self._moved_text())
        if self.start_only:
            lines.append('Never moved, since they enter the model only through its initial state: '
                         '%s.' % ', '.join(self.start_only))
        stopped = [(r, self._run_stop_text(r)) for r in results if self._run_stop_text(r)]
        times = results[0].sample.forecast_times
        if not combined:
            lines.append('The combined parameters and predicted counts were not written: the runs '
                         'assimilated different numbers of rows (%s).'
                         % ', '.join('run %d: %d' % (r.run, r.assimilated) for r in results))
        if combined and times and all(r.sample.forecasts is not None for r in results):
            lines.append('Forecast: %d interval(s), to t = %g.' % (len(times), times[-1]))
        elif self.forecast_intervals and stopped:
            lines.append('No forecast was written.')
        printed = lines[2:]
        if self.stop_reason:
            lines.append('Stopped early: %s' % self.stop_reason)
        for r, text in stopped:
            if r.degenerate:
                lines.append('Stopped early, run %d: %s' % (r.run, text))
        return lines, printed

    def _write_summary(self, folder, summary, console=True):
        """Write summary.txt in ``folder``, log it, and print the console's lines."""
        lines, printed = summary
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / 'summary.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        for line in lines:
            logger.info(line)
        for line in printed if console else ():
            print1('Liu–West filter: ' + line)

    def _distinct_start_draws(self, pop):
        columns = [j for j, v in enumerate(self.variables) if v.name in self.start_only]
        live = pop.u[pop.weights > 0.0][:, columns]
        return int(np.unique(live, axis=0).shape[0])


# --------------------------------------------------------------------------------------------
# Load-time resolution against the configuration's objects
# --------------------------------------------------------------------------------------------

def _resolve_observation(config):
    """``(column, experiment, data)`` of the one experiment on the one model, refusing what the
    raw-configuration checks could not see."""
    model_name, = config.models
    if config.constraints:
        named = sorted(c.base_suffix for c in config.constraints)
        raise _refusal('the constraints %s on the model %s' % (', '.join(named), model_name),
                       'the weights score the data rows alone')
    (suffix, data), = config.exp_data[model_name].items()
    if data.indvar != 'time':
        raise _refusal("the experiment %s, whose independent variable is '%s'" % (suffix, data.indvar),
                       'it assimilates a time course')
    columns = [c for c in sorted(data.cols, key=data.cols.get) if c != data.indvar]
    if len(columns) != 1:
        raise _refusal('%d data columns (%s) in the experiment %s' % (len(columns), ', '.join(columns),
                                                                     suffix),
                       'it scores one count per row')
    return columns[0], suffix, data


def _resolve_grid(data, suffix):
    """The row times and their step: every row one step after the one before it, the first one
    step after t = 0."""
    times = np.asarray(data.data[:, data.cols[data.indvar]], dtype=float)
    bad = [i + 1 for i, t in enumerate(times) if not np.isfinite(t) or t <= 0.0]
    if bad:
        raise PybnfError(
            'Row(s) %s of the experiment %s have the time %s. Every particle starts at t = 0, so '
            'every time must be a finite number after 0.'
            % (', '.join(map(str, bad[:6])), suffix, ', '.join('%g' % times[i - 1] for i in bad[:6])))
    gaps = np.diff(np.concatenate(([0.0], times)))
    if np.any(gaps <= 0.0):
        row = int(np.flatnonzero(gaps <= 0.0)[0]) + 1
        raise PybnfError('The times of the experiment %s do not increase at row %d (t = %g).'
                         % (suffix, row, times[row - 1]))
    step = float(np.min(gaps))
    off = np.flatnonzero(np.abs(gaps - step) > _STEP_TOL * step)
    if off.size:
        row = int(off[0]) + 1
        raise PybnfError(
            'Row %d of the experiment %s (t = %g) is %g after %s, and the step of the data is %g; '
            'each count is scored over the interval since the row before it.'
            % (row, suffix, times[row - 1], gaps[row - 1],
               'the origin t = 0' if row == 1 else 'the row before it', step),
            hint='Write a missing interval as a row holding nan, so every row is one step after '
                 'the one before it and the first is one step after 0.')
    return times, step


def _resolve_counts(data, column, suffix):
    """Every count a finite number of at least 0, or ``nan`` for an interval that is not scored."""
    counts = np.asarray(data.data[:, data.cols[column]], dtype=float)
    bad = [i + 1 for i, c in enumerate(counts)
           if not (np.isnan(c) or (np.isfinite(c) and c >= 0.0))]
    if bad:
        raise PybnfError(
            "Row(s) %s of the experiment %s have the count %s in '%s'. The negative binomial "
            'scores a count of 0 or more, so every count must be a finite number of at least 0.'
            % (', '.join(map(str, bad[:6])), suffix,
               ', '.join('%g' % counts[i - 1] for i in bad[:6]), column),
            hint='Write an interval whose count is missing as a row holding nan, which is '
                 'integrated over but not scored.')


def _resolve_noise_model(obj, column, cf):
    """The dispersion source of the one supported observation model, refusing every other."""
    others = sorted(set(obj.overrides) - {column})
    if others:
        raise _refusal("noise_model lines for %s" % ', '.join(others),
                       "the only data column is '%s'" % column)
    why = 'the count is declared by the two lines %s' % _declaration(column)
    family, sources = obj._spec_for(column)
    if not isinstance(family, NegBinomial):
        raise _refusal('the %s noise model on %s' % (type(family).__name__, column), why)
    if family.location is not MEAN:
        raise _refusal('the median location on %s' % column, why)
    source = sources.get('dispersion')
    if not isinstance(source, (FreeParameterSigma, ConstantSigma)):
        raise _refusal('the dispersion source %s on %s' % (type(source).__name__, column), why)
    if not obj._is_cumulative(column):
        raise _refusal('the column %s without the cumulative flag' % column, why)
    _check_whole_fit_objective(obj, column, source, cf)
    return source


def _check_whole_fit_objective(obj, column, source, cf):
    """Refuse a whole-fit objective that says anything but the column's line: edition 2 requires
    one, and the weights never read it, so another would be accepted and ignored."""
    family, dispersion = obj.noise, obj._default_sources().get('dispersion')
    if not isinstance(family, NegBinomial):
        differs = 'a %s noise model' % type(family).__name__
    elif family.location is not MEAN:
        differs = 'the median location'
    elif type(dispersion) is not type(source):
        differs = 'the dispersion source %s' % type(dispersion).__name__
    elif isinstance(source, FreeParameterSigma) and dispersion.name != source.name:
        differs = "the dispersion %s, where the %s line's is %s" % (dispersion.name, column,
                                                                    source.name)
    elif isinstance(source, ConstantSigma) and dispersion.const != source.const:
        differs = "the dispersion fix_at %g, where the %s line's is fix_at %g" % (
            dispersion.const, column, source.const)
    else:
        return
    what = ("the whole-fit objective 'objective = %s'" % cf['objective']
            if cf.get('objective') is not None else 'the whole-fit noise_model line')
    raise _refusal('%s (%s)' % (what, differs), 'the weights never read it',
                   hint='Declare the count with the two lines %s.'
                        % _declaration(column, _dispersion_text(source)))


def _refuse_cluster_flags(cf):
    """Refuse the -t and -s flags, which main() writes into the configuration after it loads:
    the filter connects to no cluster of the user's."""
    for flag, key in (('-t / --cluster_type', 'cluster_type'),
                      ('-s / --scheduler_file', 'scheduler_file')):
        value = cf.get(key)
        if value:
            raise _refusal('%s %s on the command line' % (flag, value),
                           'it connects to no cluster of the user\'s',
                           hint='Run without %s.' % flag.split(' / ')[0])


def _check_parameters(config):
    """A prior with finite support needs reflecting bounds to keep the kernel in it; no free
    parameter declares a start point. A ``var`` or ``logvar`` line, or a record with no prior, is
    refused upstream by the declaration rule (ADR-0118), since the filter is not a refiner."""
    for v in config.variables:
        lo, hi = v.prior_support()
        box_lo, box_hi = _box_in_sampling_space(v)
        if (np.isfinite(lo) and not np.isfinite(box_lo)) or (np.isfinite(hi) and not np.isfinite(box_hi)):
            raise _refusal('the parameter %s, whose prior has finite support but no reflecting '
                           'bounds (the u flag)' % v.name,
                           'the kernel would move it out of the support')
    if config.start_point:
        name = sorted(config.start_point)[0]
        raise _refusal('a start point for %s' % name, 'the initial population is a prior draw')
    if config.config.get('initialization') not in ('lh', 'rand'):
        raise PybnfError("initialization = %s is not a way to draw the initial population: it "
                         "must be lh or rand." % config.config.get('initialization'))


def _box_in_sampling_space(v):
    """A parameter's reflecting box in sampling space; an open side is infinite, as it is for
    :meth:`FreeParameter._reflect <pybnf.pset.FreeParameter>`."""
    if not v.bounded:
        return -np.inf, np.inf

    def side(theta):
        if not np.isfinite(theta):
            return theta
        if v.log_space and theta <= 0.0:
            return -np.inf
        return float(v.to_sampling_space(theta))

    return side(v.lower_bound), side(v.upper_bound)


def _jitter_lines(entries, variables):
    """The ``lwf_parameter_jitter`` lines as ``{name: h}``, each name a free parameter."""
    names = [v.name for v in variables]
    given = {}
    for name, h in entries:
        if name not in names:
            raise PybnfError(
                'lwf_parameter_jitter names %s, which is not a free parameter; the free '
                'parameters are %s.' % (name, ', '.join(names)))
        given[name] = float(h)
    return given


def _state_paths(given, output_dir, simulation_dir, runs):
    """Each independent run's state file: ``lwf_state_file``, or :data:`DEFAULT_STATE_FILE` in
    ``output_dir``, with ``_<r>`` before the extension for run r of several. A path main() deletes
    before every run (:mod:`pybnf.run_directories`) is refused: a continuation would not find it.
    Paths are compared with their folders resolved, so a symbolic link does not hide that one is
    inside the other."""
    base = os.path.abspath(given if given else os.path.join(output_dir, DEFAULT_STATE_FILE))
    stem, ext = os.path.splitext(base)
    paths = [base] if runs == 1 else ['%s_%d%s' % (stem, r, ext) for r in range(runs)]
    folders = [os.path.join(output_dir, d) for d in RUN_FOLDERS]
    if simulation_dir:
        folders.append(os.path.join(simulation_dir, SIMULATION_FOLDER))
    # main() deletes a file by its name, so the name of neither file is resolved.
    files = {os.path.join(os.path.realpath(output_dir), f): os.path.join(output_dir, f)
             for f in RUN_FILES}
    for path in paths:
        location = os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))
        deleted = ['inside ' + f for f in folders
                   if _within(location, os.path.realpath(f))]
        deleted += ['the file ' + files[location]] if location in files else []
        if deleted:
            raise _refusal('lwf_state_file = %s' % (given or path),
                           'the state file %s would be %s, which main() deletes before every new '
                           'run, so a continuation would not find it' % (path, deleted[0]),
                           hint='Point lwf_state_file outside the folders and files main() '
                                'deletes.')
    return paths


def _within(path, folder):
    """Whether ``path`` is ``folder`` or inside it, compared component by component."""
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:      # on another drive
        return False


def _refuse_replacing_other_files(paths, given):
    """Refuse, before a fresh run starts, to replace a file that is not a state file (the job's
    own data, say); a state file an earlier run wrote is replaced."""
    for path in paths:
        if not os.path.lexists(path):
            continue
        problem = _not_a_state_file(path)
        if problem:
            raise _refusal('lwf_state_file = %s' % (given or path),
                           'the file %s exists and %s, and a fresh run would replace it'
                           % (path, problem),
                           hint='Point lwf_state_file at a new file or at a state file an '
                                'earlier run wrote.')


def _not_a_state_file(path):
    """Why the file at ``path`` is not a state file of any format version, or None."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if 'format' not in archive.files:
                return 'is a numpy archive with no format entry'
            if str(archive['format']) != STATE_FORMAT:
                return 'is a numpy archive whose format is not %s' % STATE_FORMAT
    except OSError as exc:
        return 'cannot be read (%s)' % (exc.strerror or exc)
    except Exception:
        # numpy's own message for a text file speaks of pickled data, which would mislead.
        return 'is not a numpy archive'
    return None


def _write_state(path, arrays):
    """Write ``arrays`` to ``path`` through a synced temporary file renamed over it, so an
    interrupted write leaves the file there before whole. Uncompressed: it is written every row."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    temporary = '%s.%d.tmp' % (path, os.getpid())
    try:
        with open(temporary, 'wb') as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary)
        raise


def _read_state(path):
    """Every array of a state file, read without unpickling anything."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _network_digest(lines, free):
    """The sha256 of a model's network without its comment lines (BioNetGen's banner names its
    version) and with each free parameter's placeholder value masked."""
    digest = hashlib.sha256()
    in_parameters = False
    for line in lines:
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        if text.startswith('begin parameters'):
            in_parameters = True
        elif text.startswith('end parameters'):
            in_parameters = False
        elif in_parameters:
            parts = text.split()
            at = 1 if parts[0].isdigit() else 0
            if len(parts) > at + 1 and parts[at] in free:
                parts[at + 1] = '<free>'
                text = ' '.join(parts)
        digest.update(text.encode('utf-8') + b'\n')
    return digest.hexdigest()


def _prior_record(v):
    """A free parameter's prior, bounds and scale, as the fingerprint quotes them."""
    def num(x):
        return 'none' if x is None else repr(float(x))
    return '%s %s(%s, %s, %s) truncated to [%s, %s], bounds [%s, %s]%s, %s scale' % (
        v.name, v.type, num(v.p1), num(v.p2), num(v.p3), num(v.trunc_lb), num(v.trunc_ub),
        num(v.lower_bound), num(v.upper_bound), ' reflecting' if v.bounded else '',
        v.scale_name)


def _one_row(data, k):
    """Row ``k`` of the experiment's data as a one-row :class:`~pybnf.data.Data`."""
    headers = [data.headers[i] for i in range(len(data.headers))]
    return Data.from_columns(np.array(data.data[k:k + 1], dtype=float), headers,
                             indvar=data.indvar)


def _time_headers(times):
    """Column headers for times, wide enough that distinct times stay distinct."""
    return ['%.15g' % t for t in times]


def _write_table(path, notes, header, rows):
    """A tab-separated table whose comment lines state what the output is."""
    rows = np.asarray(rows, dtype=float)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# %s\n' % SAMPLE_STATEMENT)
        for note in notes:
            f.write('# %s\n' % note)
        f.write('\t'.join(header) + '\n')
        if rows.size:
            np.savetxt(f, rows, fmt='%.17g', delimiter='\t')
