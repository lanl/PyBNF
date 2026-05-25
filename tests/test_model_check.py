"""
Orchestration tests for ``ModelCheck`` (``fit_type='check'``) — the no-free-param
"just score the model(s) as written" path. ``ModelCheck`` does *not* subclass
``Algorithm``; ``pybnf.py`` special-cases it to run without a Cluster
(``pybnf.py``: ``if fit_type != 'check': ... else: alg.run_check(debug)``).

These are orchestration tests: the contracts are *which branch fires for which
result/config* and *which downstream call is made with what args* — not
simulation values. ModelCheck is glue over ``Job``/``run_job``/objective/
constraints, so the numerical pieces are substituted:

  * **``algorithms.run_job``** — monkeypatched to return a controlled ``Result``
    or a real ``FailedSimulation``, so the branch under test is driven directly.
  * **``algorithms.Job``** — replaced by a recorder so we assert the construction
    args without needing a simulation backend.
  * **objective / ``ConstraintCounter``** — spy fakes recording call args and
    returning canned scores / fail counts.
  * **filesystem** — real, under ``tmp_path`` (only ``__init__``'s mkdir).

``run_check`` instances are built via ``object.__new__`` with a lightweight stub
config (the real ``__init__`` is exercised separately); the ``__init__`` tests
use the real constructor with a stub config + fake models dict.

A note on the objective call (``run_check``):
``self.objective.evaluate_multiple(result.simdata, self.exp_data, result.pset,
self.config.constraints)`` uses the **explicit 4-arg convention** matching
``Algorithm.run`` (``ObjectiveFunction.evaluate_multiple(sim, exp, pset,
constraints=())``) — the result's empty ``'check'`` PSet in the pset slot,
constraints in their own slot. (It previously passed constraints positionally in
the pset slot, relying on an ``AttributeError`` fallback in ``evaluate_multiple``;
that was correct but brittle and has since been made explicit. The score is
unchanged: an empty pset means the fallback loop never ran anyway.)
``test_objective_call_uses_explicit_four_arg_convention`` pins the call shape.
"""
import logging
import os

import pytest

from .context import algorithms, pset


# --------------------------------------------------------------------------- #
# Fakes / spies
# --------------------------------------------------------------------------- #
class _SpyResult:
    """A controlled scored Result stand-in (NOT a FailedSimulation, so the
    isinstance check is False). Records normalize/postprocess calls; can be told
    to raise from postprocess to drive the post-processing-failure branch."""

    def __init__(self, simdata=None, postprocess_raises=None, ps=None):
        self.simdata = simdata if simdata is not None else {'m': {'s': object()}}
        # run_job stamps the job's (empty, 'check'-named) PSet onto the Result;
        # run_check forwards it as the pset arg of evaluate_multiple.
        self.pset = ps if ps is not None else pset.PSet([])
        self.score = None  # set by run_check from objective.evaluate_multiple
        self.normalize_calls = []
        self.postprocess_calls = []
        self._postprocess_raises = postprocess_raises

    def normalize(self, settings):
        self.normalize_calls.append(settings)

    def postprocess_data(self, settings):
        self.postprocess_calls.append(settings)
        if self._postprocess_raises is not None:
            raise self._postprocess_raises


class _SpyObjective:
    """Records evaluate_multiple call args and returns a canned score."""

    def __init__(self, score=2.5):
        self._score = score
        self.calls = []

    def evaluate_multiple(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._score


class _FakeCset:
    """A ConstraintSet stand-in: ``len(self.constraints)`` feeds the 'out of N'
    total; output_itemized_eval records its (simdata, output_dir) args."""

    def __init__(self, n):
        self.constraints = list(range(n))
        self.itemized_calls = []

    def output_itemized_eval(self, simdata, output_dir):
        self.itemized_calls.append((simdata, output_dir))


def _patch_job(monkeypatch):
    """Replace algorithms.Job with a recorder. Returns the list of (args, kwargs)
    each Job(...) was constructed with; the recorder's return value is the 'job'
    handed to run_job."""
    jobs = []

    def fake_job(*args, **kwargs):
        rec = {'args': args, 'kwargs': kwargs}
        jobs.append(rec)
        return rec
    monkeypatch.setattr(algorithms, 'Job', fake_job)
    return jobs


def _patch_run_job(monkeypatch, result):
    """Replace algorithms.run_job to return ``result`` and record its call args."""
    calls = []

    def fake_run_job(job, debug, failed_logs_dir):
        calls.append((job, debug, failed_logs_dir))
        return result
    monkeypatch.setattr(algorithms, 'run_job', fake_run_job)
    return calls


def _patch_counter(monkeypatch, fail_count):
    """Replace algorithms.ConstraintCounter with a fake whose evaluate_multiple
    returns ``fail_count`` and records its call args. Returns a holder dict that
    gets the constructed instance under 'inst'."""
    holder = {}

    class _Counter:
        def __init__(self):
            self.calls = []
            holder['inst'] = self

        def evaluate_multiple(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return fail_count
    monkeypatch.setattr(algorithms, 'ConstraintCounter', _Counter)
    return holder


def _make_check(*, objective=None, constraints=(), normalization=None,
                postprocessing=None, stochastic_seed='auto', wall_time_sim=None,
                sim_dir='/sim', model_list=None, exp_data=None):
    """Build a run_check-ready ModelCheck via object.__new__ with only the
    attributes run_check reads."""
    mc = object.__new__(algorithms.ModelCheck)
    cfg = type('Cfg', (), {})()
    cfg.config = {'normalization': normalization, 'wall_time_sim': wall_time_sim,
                  'stochastic_seed': stochastic_seed}
    cfg.postprocessing = postprocessing if postprocessing is not None else {}
    cfg.constraints = constraints
    mc.config = cfg
    mc.exp_data = exp_data if exp_data is not None else {'m': {'s': object()}}
    mc.objective = objective if objective is not None else _SpyObjective()
    mc.sim_dir = sim_dir
    mc.model_list = model_list if model_list is not None else ['MODEL']
    return mc


# =========================================================================== #
# __init__
# =========================================================================== #
class _StubModel:
    """A deepcopy-able stand-in for a Model with a mutable attribute, so the
    deepcopy contract (mutating the copy doesn't touch config.models) is testable."""

    def __init__(self, name):
        self.name = name
        self.tag = 'orig'


def _make_config_for_init(tmp_path, *, simulation_dir='', models=None):
    cfg = type('Cfg', (), {})()
    out = str(tmp_path / 'out')
    cfg.config = {'output_dir': out, 'simulation_dir': simulation_dir,
                  'wall_time_sim': None, 'stochastic_seed': 'auto',
                  'normalization': None}
    cfg.exp_data = {'m': {'s': 'EXP'}}
    cfg.obj = _SpyObjective()
    cfg.postprocessing = {}
    cfg.constraints = []
    cfg.models = models if models is not None else {'m': _StubModel('m')}
    return cfg


def test_init_sim_dir_from_simulation_dir_when_set(tmp_path):
    """When config['simulation_dir'] is set, sim_dir is '<simulation_dir>/Simulations'
    (the configured simulation tree wins over output_dir)."""
    cfg = _make_config_for_init(tmp_path, simulation_dir='/custom/sims')
    mc = algorithms.ModelCheck(cfg)
    assert mc.sim_dir == '/custom/sims/Simulations'


def test_init_sim_dir_from_output_dir_when_unset(tmp_path):
    """With simulation_dir falsy, sim_dir falls back to '<output_dir>/Simulations'."""
    cfg = _make_config_for_init(tmp_path, simulation_dir='')
    mc = algorithms.ModelCheck(cfg)
    assert mc.sim_dir == cfg.config['output_dir'] + '/Simulations'


def test_init_creates_output_dir_when_missing(tmp_path):
    """__init__ mkdir's output_dir if it doesn't already exist."""
    cfg = _make_config_for_init(tmp_path)
    assert not os.path.isdir(cfg.config['output_dir'])
    algorithms.ModelCheck(cfg)
    assert os.path.isdir(cfg.config['output_dir'])


def test_init_tolerates_existing_output_dir(tmp_path):
    """A pre-existing output_dir is left as-is (mkdir is guarded by isdir), and a
    sentinel file inside it survives — the directory is not clobbered."""
    cfg = _make_config_for_init(tmp_path)
    os.mkdir(cfg.config['output_dir'])
    sentinel = os.path.join(cfg.config['output_dir'], 'keep.txt')
    open(sentinel, 'w').close()
    algorithms.ModelCheck(cfg)  # must not raise (no FileExistsError)
    assert os.path.isfile(sentinel)


def test_init_model_list_is_independent_deepcopy(tmp_path):
    """model_list is a deepcopy of config.models' values: it carries the same
    models (by name), but mutating a copy does not touch config.models."""
    cfg = _make_config_for_init(tmp_path, models={'m1': _StubModel('m1'),
                                                  'm2': _StubModel('m2')})
    mc = algorithms.ModelCheck(cfg)
    assert sorted(m.name for m in mc.model_list) == ['m1', 'm2']
    mc.model_list[0].tag = 'mutated'
    assert all(m.tag == 'orig' for m in cfg.models.values())


def test_init_wires_objective_expdata_and_bootstrap(tmp_path):
    """__init__ reads objective/exp_data off config and fixes bootstrap_number=None
    (ModelCheck never bootstraps)."""
    cfg = _make_config_for_init(tmp_path)
    mc = algorithms.ModelCheck(cfg)
    assert mc.objective is cfg.obj
    assert mc.exp_data is cfg.exp_data
    assert mc.bootstrap_number is None


# =========================================================================== #
# run_check: job construction + run_job delegation
# =========================================================================== #
def test_builds_job_and_delegates_to_run_job(monkeypatch):
    """run_check builds a Job(model_list, empty-PSet-named-'check', 'check',
    sim_dir, wall_time_sim, None, None, {}, delete_folder=False,
    stochastic_seed_policy=config['stochastic_seed']) and hands it to run_job
    along with (debug, sim_dir) as the failed-logs directory."""
    jobs = _patch_job(monkeypatch)
    # FailedSimulation short-circuits after run_job, keeping this test about
    # construction/delegation only.
    fs = algorithms.FailedSimulation(pset.PSet([]), 'check', 1)
    calls = _patch_run_job(monkeypatch, fs)

    mc = _make_check(sim_dir='/sd', wall_time_sim=42, stochastic_seed='honorbngl',
                     model_list=['MA', 'MB'])
    mc.run_check(debug=True)

    assert len(jobs) == 1
    args, kwargs = jobs[0]['args'], jobs[0]['kwargs']
    empty = args[1]
    assert args[0] == ['MA', 'MB']               # model_list
    assert isinstance(empty, pset.PSet) and len(empty) == 0 and empty.name == 'check'
    assert args[2] == 'check'                     # job_id
    assert args[3] == '/sd'                       # sim_dir
    assert args[4] == 42                          # wall_time_sim
    assert args[5] is None and args[6] is None    # norm/postproc futures
    assert args[7] == dict()                      # job group dir
    assert kwargs == {'delete_folder': False, 'stochastic_seed_policy': 'honorbngl'}

    # run_job got the constructed job, the debug flag, and sim_dir as failed-logs dir.
    assert len(calls) == 1
    job_arg, debug_arg, failed_logs = calls[0]
    assert job_arg is jobs[0] and debug_arg is True and failed_logs == '/sd'


def test_debug_flag_forwarded_to_run_job(monkeypatch):
    """The debug flag passed to run_check is forwarded to run_job unchanged."""
    _patch_job(monkeypatch)
    calls = _patch_run_job(monkeypatch,
                           algorithms.FailedSimulation(pset.PSet([]), 'check', 1))
    _make_check().run_check(debug=False)
    assert calls[0][1] is False


# =========================================================================== #
# run_check: FailedSimulation branch
# =========================================================================== #
def test_failed_simulation_prints_and_returns_early(monkeypatch, capsys):
    """A FailedSimulation result ⇒ 'Simulation failed.' is printed and run_check
    returns immediately: the objective is never evaluated."""
    _patch_job(monkeypatch)
    _patch_run_job(monkeypatch,
                   algorithms.FailedSimulation(pset.PSet([]), 'check', 1))
    obj = _SpyObjective()
    mc = _make_check(objective=obj)

    mc.run_check()

    assert 'Simulation failed.' in capsys.readouterr().out
    assert obj.calls == []  # no scoring after a failed sim


# =========================================================================== #
# run_check: success happy path (no constraints)
# =========================================================================== #
def test_success_normalizes_postprocesses_scores_and_prints(monkeypatch, capsys):
    """Happy path with no constraints: normalize(config['normalization']) and
    postprocess_data(config.postprocessing) run on the result, the objective
    scores it (result.score set to the return), the objective value is printed,
    and the constraint block is skipped."""
    _patch_job(monkeypatch)
    result = _SpyResult()
    _patch_run_job(monkeypatch, result)
    obj = _SpyObjective(score=7.5)
    postproc = {('m', 's'): 'script.py'}
    mc = _make_check(objective=obj, constraints=(), normalization='init',
                     postprocessing=postproc)

    mc.run_check()

    assert result.normalize_calls == ['init']
    assert result.postprocess_calls == [postproc]
    assert result.score == 7.5
    assert len(obj.calls) == 1
    assert 'Objective value is 7.5' in capsys.readouterr().out


def test_objective_call_uses_explicit_four_arg_convention(monkeypatch):
    """run_check calls objective.evaluate_multiple(simdata, exp_data, result.pset,
    constraints) — the explicit four-arg convention matching Algorithm.run: the
    result's (empty 'check') PSet in the pset slot and constraints in their own
    slot. This is what makes constraint penalties fold into the score without
    relying on the AttributeError fallback (see module docstring)."""
    _patch_job(monkeypatch)
    result = _SpyResult()
    _patch_run_job(monkeypatch, result)
    obj = _SpyObjective(score=1.0)
    constraints = (_FakeCset(1),)
    exp = {'m': {'s': object()}}
    mc = _make_check(objective=obj, constraints=constraints, exp_data=exp)
    _patch_counter(monkeypatch, fail_count=0)  # constraints non-empty -> counter runs

    mc.run_check()

    (args, kwargs), = obj.calls
    assert kwargs == {}
    assert len(args) == 4
    assert args[0] is result.simdata
    assert args[1] is exp
    assert args[2] is result.pset      # pset slot carries the result's own pset
    assert args[3] is constraints      # constraints in their own (4th) slot


# =========================================================================== #
# run_check: post-processing failure branch
# =========================================================================== #
def test_postprocess_failure_logs_and_returns_before_scoring(monkeypatch, capsys, caplog):
    """If postprocess_data raises, run_check logs the exception, prints the
    'post-processing script failed' notice, and returns before scoring — the
    objective is never evaluated."""
    _patch_job(monkeypatch)
    result = _SpyResult(postprocess_raises=ValueError('bad script'))
    _patch_run_job(monkeypatch, result)
    obj = _SpyObjective()
    mc = _make_check(objective=obj)

    with caplog.at_level(logging.ERROR, logger='pybnf.algorithms'):
        mc.run_check()

    assert 'post-processing script failed' in capsys.readouterr().out
    assert any('post-processing script failed' in r.message for r in caplog.records)
    assert obj.calls == []  # scoring skipped after a post-processing failure


# =========================================================================== #
# run_check: None-score branch
# =========================================================================== #
def test_none_score_prints_nan_message_and_skips_constraints(monkeypatch, capsys):
    """A None score (NaN/Inf in the simulation) ⇒ the NaN/Inf message is printed
    and run_check returns before the constraint block — ConstraintCounter is never
    constructed."""
    _patch_job(monkeypatch)
    _patch_run_job(monkeypatch, _SpyResult())
    holder = _patch_counter(monkeypatch, fail_count=0)
    mc = _make_check(objective=_SpyObjective(score=None),
                     constraints=(_FakeCset(2),))

    mc.run_check()

    out = capsys.readouterr().out
    assert 'NaN or Inf' in out
    assert 'Objective value' not in out  # never reached the print
    assert 'inst' not in holder           # constraint block skipped


# =========================================================================== #
# run_check: constraint reporting block
# =========================================================================== #
def test_no_constraints_skips_constraint_block(monkeypatch, capsys):
    """With an empty constraint list, the constraint-counting block is skipped:
    no ConstraintCounter, no 'Satisfied ... out of ...' line."""
    _patch_job(monkeypatch)
    _patch_run_job(monkeypatch, _SpyResult())
    holder = _patch_counter(monkeypatch, fail_count=0)
    _make_check(objective=_SpyObjective(score=1.0), constraints=()).run_check()

    assert 'Satisfied' not in capsys.readouterr().out
    assert 'inst' not in holder  # ConstraintCounter never constructed


def test_constraints_counted_reported_and_itemized(monkeypatch, capsys):
    """With constraints present: ConstraintCounter.evaluate_multiple is called with
    (simdata, exp_data, constraints), 'Satisfied (total-fail_count) out of total'
    is printed (total = Σ len(cset.constraints)), and each cset's
    output_itemized_eval is invoked with (simdata, sim_dir)."""
    _patch_job(monkeypatch)
    result = _SpyResult()
    _patch_run_job(monkeypatch, result)
    holder = _patch_counter(monkeypatch, fail_count=2)
    csets = (_FakeCset(3), _FakeCset(2))  # total = 5 constraints
    exp = {'m': {'s': object()}}
    mc = _make_check(objective=_SpyObjective(score=1.0), constraints=csets,
                     exp_data=exp, sim_dir='/the/sim/dir')

    mc.run_check()

    # Counter called with the legacy (simdata, exp, constraints) positional triple.
    (cargs, ckwargs), = holder['inst'].calls
    assert cargs == (result.simdata, exp, csets) and ckwargs == {}
    # total - fail_count = 5 - 2 = 3 satisfied.
    assert 'Satisfied 3 out of 5 constraints' in capsys.readouterr().out
    # Each cset itemized against (simdata, sim_dir).
    for cset in csets:
        assert cset.itemized_calls == [(result.simdata, '/the/sim/dir')]
