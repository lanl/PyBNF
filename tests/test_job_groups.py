"""
Unit tests for JobGroup, MultimodelJobGroup, HybridJobGroup, Result.add_result(), and
the make_job() branching logic (issue #49 pre-implementation tests).
"""

from .context import data, algorithms, pset, config
from unittest.mock import patch
import numpy as np
import numpy.testing as npt
import os
import shutil


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(values):
    """Create a Data object with the given 2D array and standard column layout."""
    d = data.Data()
    d.cols = {'time': 0, 'v1_result': 1, 'v2_result': 2}
    d.data = np.array(values, dtype=float)
    return d


def _make_result(name, model_data):
    """
    Create a Result with given name and simdata dict.
    model_data: dict of {model_name: {suffix: Data}}
    """
    p = pset.PSet([pset.FreeParameter('v1__FREE', 'uniform_var', 0, 10, 5.0)])
    res = algorithms.Result(p, model_data, name)
    res.score = 42.0
    return res


def _make_failed(name):
    """Create a FailedSimulation result."""
    p = pset.PSet([pset.FreeParameter('v1__FREE', 'uniform_var', 0, 10, 5.0)])
    return algorithms.FailedSimulation(p, name, 1)


_no_bng = patch.object(config.Configuration, '_load_simulators')
_no_init = patch.object(algorithms.Algorithm, '_initialize_models', return_value=[])

_BASE_CONFIG = {
    ('uniform_var', 'v1__FREE'): [0, 10],
    ('uniform_var', 'v2__FREE'): [0, 10],
    'models': {'tests/bngl_files/parabola.bngl'},
    'exp_data': {'tests/bngl_files/par1.exp'},
    'tests/bngl_files/parabola.bngl': ['tests/bngl_files/par1.exp'],
}


def _make_algo(extra_config):
    """Create a PSO algorithm with mocked BNG for testing make_job()."""
    cfg_dict = dict(_BASE_CONFIG)
    cfg_dict.update(extra_config)
    out_dir = cfg_dict.get('output_dir', 'test_jg')
    os.makedirs(os.path.join(out_dir, 'Results'), exist_ok=True)
    with _no_bng, _no_init:
        cfg = config.Configuration(cfg_dict)
        algo = algorithms.ParticleSwarm(cfg)
    return algo


def _cleanup():
    for d in ['test_jg', 'test_jg_smooth', 'test_jg_par', 'test_jg_hybrid', 'test_jg_default']:
        if os.path.isdir(d):
            shutil.rmtree(d)


# ===========================================================================
# Tests: Result.add_result()
# ===========================================================================

class TestResultAddResult:

    def test_merge_disjoint_models(self):
        """add_result merges simdata from two results with different model keys."""
        data_a = _make_data([[1.0, 2.0, 3.0]])
        data_b = _make_data([[4.0, 5.0, 6.0]])

        res1 = _make_result('job1', {'modelA': {'suf1': data_a}})
        res2 = _make_result('job2', {'modelB': {'suf2': data_b}})

        res1.add_result(res2)

        assert 'modelA' in res1.simdata
        assert 'modelB' in res1.simdata
        npt.assert_array_equal(res1.simdata['modelA']['suf1'].data, [[1.0, 2.0, 3.0]])
        npt.assert_array_equal(res1.simdata['modelB']['suf2'].data, [[4.0, 5.0, 6.0]])

    def test_merge_overwrites_duplicate_keys(self):
        """add_result uses dict.update, so duplicate model keys get overwritten."""
        data_a = _make_data([[1.0, 2.0, 3.0]])
        data_b = _make_data([[9.0, 9.0, 9.0]])

        res1 = _make_result('job1', {'model': {'suf': data_a}})
        res2 = _make_result('job2', {'model': {'suf': data_b}})

        res1.add_result(res2)
        npt.assert_array_equal(res1.simdata['model']['suf'].data, [[9.0, 9.0, 9.0]])

    def test_merge_into_empty(self):
        """add_result into an empty simdata dict."""
        data_a = _make_data([[1.0, 2.0, 3.0]])
        res1 = _make_result('job1', {})
        res2 = _make_result('job2', {'model': {'suf': data_a}})

        res1.add_result(res2)
        assert 'model' in res1.simdata


# ===========================================================================
# Tests: JobGroup
# ===========================================================================

class TestJobGroup:

    def test_job_finished_returns_true_when_all_done(self):
        group = algorithms.JobGroup('sim_1', ['sim_1_rep0', 'sim_1_rep1', 'sim_1_rep2'])

        for i in range(2):
            res = _make_result('sim_1_rep%d' % i, {'m': {'s': _make_data([[1, 2, 3]])}})
            assert group.job_finished(res) is False

        res = _make_result('sim_1_rep2', {'m': {'s': _make_data([[1, 2, 3]])}})
        assert group.job_finished(res) is True

    def test_job_finished_immediate_on_failure(self):
        """A FailedSimulation makes the group immediately done."""
        group = algorithms.JobGroup('sim_1', ['sim_1_rep0', 'sim_1_rep1'])

        failed = _make_failed('sim_1_rep0')
        assert group.job_finished(failed) is True

    def test_job_finished_ignores_after_failure(self):
        """After a failure, subsequent results return False (group already done)."""
        group = algorithms.JobGroup('sim_1', ['sim_1_rep0', 'sim_1_rep1'])

        failed = _make_failed('sim_1_rep0')
        group.job_finished(failed)

        res = _make_result('sim_1_rep1', {'m': {'s': _make_data([[1, 2, 3]])}})
        assert group.job_finished(res) is False

    def test_average_results_computes_mean(self):
        """average_results returns element-wise mean of Data arrays."""
        group = algorithms.JobGroup('sim_1', ['sim_1_rep0', 'sim_1_rep1'])

        d0 = _make_data([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
        d1 = _make_data([[3.0, 6.0, 8.0], [4.0, 8.0, 12.0]])

        group.job_finished(_make_result('sim_1_rep0', {'m': {'s': d0}}))
        group.job_finished(_make_result('sim_1_rep1', {'m': {'s': d1}}))

        avg = group.average_results()
        assert avg.name == 'sim_1'
        npt.assert_array_almost_equal(avg.simdata['m']['s'].data,
                                      [[2.0, 4.0, 6.0], [3.0, 6.0, 10.0]])

    def test_average_results_multiple_models(self):
        """average_results handles multiple models and suffixes."""
        group = algorithms.JobGroup('sim_1', ['sim_1_rep0', 'sim_1_rep1'])

        simdata0 = {
            'modelA': {'suf1': _make_data([[1.0, 2.0, 3.0]])},
            'modelB': {'suf2': _make_data([[10.0, 20.0, 30.0]])},
        }
        simdata1 = {
            'modelA': {'suf1': _make_data([[3.0, 4.0, 5.0]])},
            'modelB': {'suf2': _make_data([[30.0, 40.0, 50.0]])},
        }

        group.job_finished(_make_result('sim_1_rep0', simdata0))
        group.job_finished(_make_result('sim_1_rep1', simdata1))

        avg = group.average_results()
        npt.assert_array_almost_equal(avg.simdata['modelA']['suf1'].data, [[2.0, 3.0, 4.0]])
        npt.assert_array_almost_equal(avg.simdata['modelB']['suf2'].data, [[20.0, 30.0, 40.0]])

    def test_average_results_returns_failed_on_failure(self):
        """average_results returns the FailedSimulation with the group's job_id."""
        group = algorithms.JobGroup('sim_1', ['sim_1_rep0', 'sim_1_rep1'])

        failed = _make_failed('sim_1_rep0')
        group.job_finished(failed)

        result = group.average_results()
        assert isinstance(result, algorithms.FailedSimulation)
        assert result.name == 'sim_1'


# ===========================================================================
# Tests: MultimodelJobGroup
# ===========================================================================

class TestMultimodelJobGroup:

    def test_job_finished_returns_true_when_all_done(self):
        group = algorithms.MultimodelJobGroup('sim_1', ['sim_1_part0', 'sim_1_part1'])

        res0 = _make_result('sim_1_part0', {'modelA': {'s': _make_data([[1, 2, 3]])}})
        assert group.job_finished(res0) is False

        res1 = _make_result('sim_1_part1', {'modelB': {'s': _make_data([[4, 5, 6]])}})
        assert group.job_finished(res1) is True

    def test_average_results_merges_models(self):
        """average_results merges disjoint model results into one Result."""
        group = algorithms.MultimodelJobGroup('sim_1', ['sim_1_part0', 'sim_1_part1'])

        res0 = _make_result('sim_1_part0', {'modelA': {'s': _make_data([[1, 2, 3]])}})
        res1 = _make_result('sim_1_part1', {'modelB': {'s': _make_data([[4, 5, 6]])}})

        group.job_finished(res0)
        group.job_finished(res1)

        merged = group.average_results()
        assert merged.name == 'sim_1'
        assert 'modelA' in merged.simdata
        assert 'modelB' in merged.simdata
        npt.assert_array_equal(merged.simdata['modelA']['s'].data, [[1, 2, 3]])
        npt.assert_array_equal(merged.simdata['modelB']['s'].data, [[4, 5, 6]])

    def test_average_results_three_partitions(self):
        """Merging three model partitions."""
        group = algorithms.MultimodelJobGroup('sim_1',
                                              ['sim_1_part0', 'sim_1_part1', 'sim_1_part2'])

        for i, name in enumerate(['mA', 'mB', 'mC']):
            res = _make_result('sim_1_part%d' % i, {name: {'s': _make_data([[float(i)]])}})
            group.job_finished(res)

        merged = group.average_results()
        assert set(merged.simdata.keys()) == {'mA', 'mB', 'mC'}

    def test_average_results_returns_failed_on_failure(self):
        group = algorithms.MultimodelJobGroup('sim_1', ['sim_1_part0', 'sim_1_part1'])

        failed = _make_failed('sim_1_part0')
        group.job_finished(failed)

        result = group.average_results()
        assert isinstance(result, algorithms.FailedSimulation)
        assert result.name == 'sim_1'


# ===========================================================================
# Tests: HybridJobGroup
# ===========================================================================

class TestHybridJobGroup:

    def test_job_finished_returns_true_when_all_parts_and_replicates_done(self):
        group = algorithms.HybridJobGroup(
            'sim_1',
            [('sim_1_rep0', ['sim_1_rep0_part0', 'sim_1_rep0_part1']),
             ('sim_1_rep1', ['sim_1_rep1_part0', 'sim_1_rep1_part1'])]
        )

        names = ['sim_1_rep0_part0', 'sim_1_rep1_part0', 'sim_1_rep0_part1']
        for name in names:
            res = _make_result(name, {'m': {'s': _make_data([[1, 2, 3]])}})
            assert group.job_finished(res) is False

        res = _make_result('sim_1_rep1_part1', {'m': {'s': _make_data([[1, 2, 3]])}})
        assert group.job_finished(res) is True

    def test_average_results_merges_models_then_averages_replicates(self):
        group = algorithms.HybridJobGroup(
            'sim_1',
            [('sim_1_rep0', ['sim_1_rep0_part0', 'sim_1_rep0_part1']),
             ('sim_1_rep1', ['sim_1_rep1_part0', 'sim_1_rep1_part1'])]
        )

        group.job_finished(_make_result(
            'sim_1_rep0_part0',
            {'modelA': {'s': _make_data([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])}}
        ))
        group.job_finished(_make_result(
            'sim_1_rep0_part1',
            {'modelB': {'s': _make_data([[10.0, 20.0, 40.0], [20.0, 40.0, 80.0]])}}
        ))
        group.job_finished(_make_result(
            'sim_1_rep1_part0',
            {'modelA': {'s': _make_data([[3.0, 6.0, 8.0], [4.0, 8.0, 12.0]])}}
        ))
        group.job_finished(_make_result(
            'sim_1_rep1_part1',
            {'modelB': {'s': _make_data([[30.0, 60.0, 80.0], [40.0, 80.0, 120.0]])}}
        ))

        avg = group.average_results()

        assert avg.name == 'sim_1'
        assert set(avg.simdata.keys()) == {'modelA', 'modelB'}
        npt.assert_array_almost_equal(avg.simdata['modelA']['s'].data,
                                      [[2.0, 4.0, 6.0], [3.0, 6.0, 10.0]])
        npt.assert_array_almost_equal(avg.simdata['modelB']['s'].data,
                                      [[20.0, 40.0, 60.0], [30.0, 60.0, 100.0]])

    def test_average_results_returns_failed_on_failure(self):
        group = algorithms.HybridJobGroup(
            'sim_1',
            [('sim_1_rep0', ['sim_1_rep0_part0', 'sim_1_rep0_part1']),
             ('sim_1_rep1', ['sim_1_rep1_part0', 'sim_1_rep1_part1'])]
        )

        failed = _make_failed('sim_1_rep0_part0')
        assert group.job_finished(failed) is True

        result = group.average_results()
        assert isinstance(result, algorithms.FailedSimulation)
        assert result.name == 'sim_1'


# ===========================================================================
# Tests: make_job() branching
# ===========================================================================

class TestMakeJob:

    @classmethod
    def setup_class(cls):
        _cleanup()

    @classmethod
    def teardown_class(cls):
        _cleanup()

    def test_default_single_job(self):
        """With smoothing=1 and parallelize_models=1, make_job returns one job."""
        algo = _make_algo({
            'population_size': 2, 'max_iterations': 5,
            'fit_type': 'pso', 'output_dir': 'test_jg_default'})

        p = algo.random_pset()
        p.name = 'test_pset'
        jobs = algo.make_job(p)

        assert len(jobs) == 1
        assert jobs[0].job_id == 'test_pset'

    def test_smoothing_creates_replicate_jobs(self):
        """With smoothing=3, make_job creates 3 jobs with a JobGroup."""
        algo = _make_algo({
            'population_size': 2, 'max_iterations': 5,
            'smoothing': 3,
            'fit_type': 'pso', 'output_dir': 'test_jg_smooth'})

        p = algo.random_pset()
        p.name = 'sim_1'
        jobs = algo.make_job(p)

        assert len(jobs) == 3
        names = [j.job_id for j in jobs]
        assert names == ['sim_1_rep0', 'sim_1_rep1', 'sim_1_rep2']

        # All jobs should map to the same JobGroup
        groups = [algo.job_group_dir[n] for n in names]
        assert all(g is groups[0] for g in groups)
        assert isinstance(groups[0], algorithms.JobGroup)
        assert not isinstance(groups[0], algorithms.MultimodelJobGroup)

    def test_parallelize_models_creates_partitioned_jobs(self):
        """With parallelize_models=2, make_job creates 2 jobs with a MultimodelJobGroup."""
        algo = _make_algo({
            'population_size': 2, 'max_iterations': 5,
            'parallelize_models': 1,  # Must be <= number of models; we have 1 model
            'fit_type': 'pso', 'output_dir': 'test_jg_par'})

        # With only 1 model, parallelize_models=1 is the max. Test the branch
        # by checking that it creates the right group type when enabled.
        # We need to artificially set parallelize_models > 1 and add fake models.
        algo.config.config['parallelize_models'] = 2
        algo.model_list = ['fake_model_A', 'fake_model_B']

        p = algo.random_pset()
        p.name = 'sim_1'
        jobs = algo.make_job(p)

        assert len(jobs) == 2
        names = [j.job_id for j in jobs]
        assert names == ['sim_1_part0', 'sim_1_part1']

        groups = [algo.job_group_dir[n] for n in names]
        assert all(g is groups[0] for g in groups)
        assert isinstance(groups[0], algorithms.MultimodelJobGroup)

    def test_smoothing_jobs_get_full_model_list(self):
        """Each smoothing replicate gets the complete model list."""
        algo = _make_algo({
            'population_size': 2, 'max_iterations': 5,
            'smoothing': 2,
            'fit_type': 'pso', 'output_dir': 'test_jg_smooth'})

        algo.model_list = ['model_A', 'model_B', 'model_C']

        p = algo.random_pset()
        p.name = 'sim_1'
        jobs = algo.make_job(p)

        for j in jobs:
            assert j.models == ['model_A', 'model_B', 'model_C']

    def test_parallelize_models_partitions_model_list(self):
        """Model list is partitioned across jobs."""
        algo = _make_algo({
            'population_size': 2, 'max_iterations': 5,
            'fit_type': 'pso', 'output_dir': 'test_jg_par'})

        algo.config.config['parallelize_models'] = 3
        algo.model_list = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']

        p = algo.random_pset()
        p.name = 'sim_1'
        jobs = algo.make_job(p)

        assert len(jobs) == 3
        # 6 models / 3 partitions = 2 each
        assert jobs[0].models == ['m1', 'm2']
        assert jobs[1].models == ['m3', 'm4']
        assert jobs[2].models == ['m5', 'm6']

    def test_combined_smoothing_and_parallelize_models_creates_hybrid_group(self):
        """Combined smoothing and model parallelism creates one HybridJobGroup."""
        algo = _make_algo({
            'population_size': 2, 'max_iterations': 5,
            'smoothing': 2,
            'fit_type': 'pso', 'output_dir': 'test_jg_hybrid'})

        algo.config.config['parallelize_models'] = 3
        algo.model_list = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']

        p = algo.random_pset()
        p.name = 'sim_1'
        jobs = algo.make_job(p)

        assert len(jobs) == 6
        names = [j.job_id for j in jobs]
        assert names == ['sim_1_rep0_part0', 'sim_1_rep0_part1', 'sim_1_rep0_part2',
                         'sim_1_rep1_part0', 'sim_1_rep1_part1', 'sim_1_rep1_part2']

        groups = [algo.job_group_dir[n] for n in names]
        assert all(g is groups[0] for g in groups)
        assert isinstance(groups[0], algorithms.HybridJobGroup)

        assert jobs[0].models == ['m1', 'm2']
        assert jobs[1].models == ['m3', 'm4']
        assert jobs[2].models == ['m5', 'm6']
        assert jobs[3].models == ['m1', 'm2']
        assert jobs[4].models == ['m3', 'm4']
        assert jobs[5].models == ['m5', 'm6']

    def test_auto_job_id_when_no_name(self):
        """When pset has no name, make_job auto-assigns sim_N."""
        algo = _make_algo({
            'population_size': 2, 'max_iterations': 5,
            'fit_type': 'pso', 'output_dir': 'test_jg_default'})

        p = algo.random_pset()
        p.name = ''
        jobs1 = algo.make_job(p)
        jobs2 = algo.make_job(p)

        # Should get sequential IDs
        assert jobs1[0].job_id != jobs2[0].job_id
