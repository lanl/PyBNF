"""
Dataflow tests for ``Job.run_simulation`` — the orchestration that turns a
parameter set into a scored ``Result``.

Existing suites cover the *failure* branches (test_failed_sim_handling.py) and the
no-scoring ``calc_future=None`` happy path with a real backend
(test_job_class.py). The gaps these tests close are the *scoring* dataflow and
the side-effect contracts, exercised with fakes so no BioNetGen/bngsim backend is
needed:

  * a successful evaluation moves the simulation output into ``res.out`` and
    clears ``res.simdata`` (a deliberate memory contract), and the scored value
    is exactly what the scattered calculator returned;
  * a ``None`` score (NaN/Inf simulation) is flagged with ``res.out = inf`` and a
    warning, rather than silently kept;
  * the per-evaluation model copy is stamped with the smoothing replicate index
    and stochastic-seed policy (the determinism plumbing);
  * a pre-existing output folder triggers a ``_rerun`` folder instead of
    clobbering;
  * ``delete_folder`` cleans up; relative ``output_dir`` is made absolute.

Substitution strategy: the model and the scattered ObjectiveCalculator are
**fakes** (Job calls each only a couple of times, through a narrow interface);
the filesystem is **real** under ``tmp_path``.
"""
import os

import numpy as np

from .context import algorithms, data, pset


def _make_pset():
    return pset.PSet([pset.FreeParameter('v1__FREE', 'uniform_var', 0, 10, 5.0)])


def _make_data():
    d = data.Data()
    d.cols = {'time': 0, 'v1_result': 1}
    d.data = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    return d


class _ModelCopy:
    """A per-evaluation model copy that records the seed context stamped onto it
    and returns canned simulation data."""

    def __init__(self, name, params):
        self.name = name
        self.params = params
        self._pybnf_replicate_index = None
        self._pybnf_stochastic_seed_policy = None

    def execute(self, folder, file_prefix, timeout):
        return {'time_course': _make_data()}


class _FakeModel:
    def __init__(self, name='m'):
        self.name = name
        self.copies = []

    def copy_with_param_set(self, params):
        c = _ModelCopy(self.name, params)
        self.copies.append(c)
        return c


class _FakeCalc:
    """Stand-in for a scattered ObjectiveCalculator future: ``.result()`` returns
    self, and ``evaluate_objective`` records the simdata it scored and returns a
    canned value."""

    def __init__(self, score):
        self._score = score
        self.scored_simdata = None

    def result(self):
        return self

    def evaluate_objective(self, simdata, ps, show_warnings=True):
        self.scored_simdata = simdata
        return self._score


def _make_job(tmp_path, model, calc, job_id='sim_1', **kwargs):
    return algorithms.Job(
        [model], _make_pset(), job_id, str(tmp_path), None,
        calc_future=calc, norm_settings=None, postproc_settings=dict(), **kwargs)


# --------------------------------------------------------------------------- #
# Scoring success: dataflow into res.out, res.simdata cleared
# --------------------------------------------------------------------------- #
def test_success_scores_and_moves_simdata_to_out(tmp_path):
    """On a successful evaluation: the scattered calculator scores the model
    output, that score lands on res.score, the simulation output is moved to
    res.out, and res.simdata is cleared to None (memory contract)."""
    model = _FakeModel()
    calc = _FakeCalc(score=2.5)
    res = _make_job(tmp_path, model, calc).run_simulation()

    assert not res.failed
    assert res.score == 2.5
    # The scorer received the model output keyed by model name -> suffix -> Data.
    assert set(calc.scored_simdata.keys()) == {'m'}
    assert 'time_course' in calc.scored_simdata['m']
    # Output handed forward as res.out; res.simdata cleared.
    assert res.out is not None and set(res.out.keys()) == {'m'}
    assert res.simdata is None


def test_none_score_flags_inf_and_warns(tmp_path, caplog):
    """A None score (simulation produced NaNs/Infs) must be flagged: res.out is
    set to inf and a warning is logged, rather than the result being kept as if
    valid."""
    model = _FakeModel()
    calc = _FakeCalc(score=None)
    with caplog.at_level('WARNING', logger='pybnf.algorithms'):
        res = _make_job(tmp_path, model, calc).run_simulation()

    assert res.score is None
    assert res.out == np.inf
    assert any('NaNs or Infs' in r.message for r in caplog.records)


def test_no_calc_future_returns_unscored_result_with_simdata(tmp_path):
    """With calc_future=None (smoothing / model-parallel jobs score later), the
    job returns an unscored Result whose simdata is intact — the scoring block is
    skipped entirely."""
    model = _FakeModel()
    res = _make_job(tmp_path, model, calc=None).run_simulation()

    assert not res.failed
    assert res.score is None
    assert res.simdata is not None and set(res.simdata.keys()) == {'m'}


# --------------------------------------------------------------------------- #
# Determinism plumbing: seed context stamped on the model copy
# --------------------------------------------------------------------------- #
def test_seed_policy_and_replicate_stamped_on_model_copy(tmp_path):
    """_run_models stamps the smoothing replicate index and the stochastic-seed
    policy onto each per-evaluation model copy, so backends can derive
    deterministic seeds without threading config through execute()."""
    model = _FakeModel()
    _make_job(tmp_path, model, calc=None,
              replicate_index=2, stochastic_seed_policy='random_honorbngl').run_simulation()

    assert len(model.copies) == 1
    copy = model.copies[0]
    assert copy._pybnf_replicate_index == 2
    assert copy._pybnf_stochastic_seed_policy == 'random_honorbngl'


# --------------------------------------------------------------------------- #
# Filesystem side effects
# --------------------------------------------------------------------------- #
def test_folder_collision_creates_rerun_folder(tmp_path):
    """If the job's output folder already exists (e.g. dask ran the job twice),
    run_simulation must fall back to a '<job_id>_rerunN' folder rather than
    clobber the existing one."""
    job = _make_job(tmp_path, _FakeModel(), calc=None, job_id='sim_1')
    os.mkdir(job.folder)  # pre-existing folder forces the rerun path

    job.run_simulation()

    assert job.folder == '%s/sim_1_rerun1' % str(tmp_path)
    assert os.path.isdir(job.folder)


def test_delete_folder_removes_simulation_folder(tmp_path):
    """With delete_folder=True the simulation folder is removed after the run."""
    job = _make_job(tmp_path, _FakeModel(), calc=None, delete_folder=True)
    job.run_simulation()
    assert not os.path.isdir(job.folder)


def test_folder_retained_by_default(tmp_path):
    """Without delete_folder the folder persists (so outputs can be inspected)."""
    job = _make_job(tmp_path, _FakeModel(), calc=None)
    job.run_simulation()
    assert os.path.isdir(job.folder)


# --------------------------------------------------------------------------- #
# Path normalization (workers don't inherit the scheduler's cwd)
# --------------------------------------------------------------------------- #
def test_relative_output_dir_made_absolute():
    """A relative output_dir is anchored to the scheduler's cwd at construction,
    because dask workers don't share the relative-path context."""
    job = algorithms.Job([_FakeModel()], _make_pset(), 'sim_1', 'rel/out', None,
                         calc_future=None, norm_settings=None, postproc_settings=dict())
    assert job.output_dir == os.getcwd() + '/rel/out'
    assert job.folder == os.getcwd() + '/rel/out/sim_1'


def test_absolute_output_dir_preserved():
    job = algorithms.Job([_FakeModel()], _make_pset(), 'sim_1', '/abs/out', None,
                         calc_future=None, norm_settings=None, postproc_settings=dict())
    assert job.output_dir == '/abs/out'
