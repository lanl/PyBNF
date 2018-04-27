from .context import data, algorithms, pset, objective, config
import numpy as np
from copy import deepcopy

from shutil import rmtree


class TestSimplex:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.data1s = [
            '# time    v1_result    v2_result    v3_result\n',
            ' 1 2.1   3.1   6.1\n',
        ]
        cls.d1s = data.Data()
        cls.d1s.data = cls.d1s._read_file_lines(cls.data1s, '\s+')

        cls.config = config.Configuration({
            'population_size': 2, 'max_iterations': 20, 'fit_type': 'sim', 'simplex_start_step': 1.0,
            'simplex_reflection': 1., 'simplex_expansion': 1., 'simplex_contraction': 0.5, 'simplex_shrink': 0.5,
            ('var', 'v1__FREE__'): [2.], ('var', 'v2__FREE__'): [3.], ('var', 'v3__FREE__'): [4.],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp']})

        cls.logconfig = config.Configuration({
            'population_size': 2, 'max_iterations': 20, 'fit_type': 'sim', 'simplex_start_step': 1.0,
            'simplex_reflection': 1., 'simplex_expansion': 1., 'simplex_contraction': 0.5, 'simplex_shrink': 0.5,
            ('logvar', 'v1__FREE__'): [2.], ('logvar', 'v2__FREE__'): [3.], ('logvar', 'v3__FREE__'): [4.],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp']})

    @classmethod
    def teardown_class(cls):
        rmtree('bnf_out')

    def test_start(self):
        sim = algorithms.SimplexAlgorithm(deepcopy(self.config))
        sim.variables.sort()  # Required for Python <= 3.5 to be sure we are checking the correct indices in the simplex
        first = sim.start_run()
        assert len(first) == 4
        assert first[0]['v1__FREE__'] == 2.
        assert first[3]['v3__FREE__'] == 5.

    def test_updates(self):
        sim = algorithms.SimplexAlgorithm(deepcopy(self.config))
        sim.variables.sort()  # Required for Python <= 3.5 to be sure we are checking the correct indices in the simplex
        first = sim.start_run()
        next_params = []
        for p, score in zip(first, [5., 7., 8., 6.]):
            res = algorithms.Result(p, self.data1s, p.name)
            res.score = score
            next_params += sim.got_result(res)

        assert len(next_params) == 2
        # First point should be the reflection of the worst PSet, where v2 += 1
        # M = (2 1/3, 3, 4 1/3)
        np.testing.assert_almost_equal(next_params[0]['v1__FREE__'], 8./3.)
        # Second point should be reflection of v1
        # M = (2, 3 1/3, 4 1/3)
        np.testing.assert_almost_equal(next_params[1]['v1__FREE__'], 1.)

        res = algorithms.Result(next_params[1], self.data1s, next_params[1].name)
        res.score = 5.5
        # Case 2 - take it, do nothing else.
        p2_1 = sim.got_result(res)
        assert p2_1 == []
        res = algorithms.Result(next_params[0], self.data1s, next_params[0].name)
        res.score = 4.5
        # Case 1 - pick an expansion point
        p2_2 = sim.got_result(res)
        assert len(p2_2) == 1
        # print(sim.centroids)
        # print(p2_2)
        np.testing.assert_almost_equal(p2_2[0]['v1__FREE__'], 3.)
        np.testing.assert_almost_equal(p2_2[0]['v2__FREE__'], 1.)

        res = algorithms.Result(p2_2[0], self.data1s, p2_2[0].name)
        res.score = 4.75
        iter_2 = sim.got_result(res)
        assert len(iter_2) == 2
        assert sim.simplex == [(4.5, next_params[0]), (5., first[0]), (5.5, next_params[1]), (6., first[3])]
        # ((2 2/3, 2, 4 2/3), (2,3,4), (1, 3 2/3, 4 2/3), (2,3,5))

        # Next iteration. This time, we'll give non-improvements, forcing a simplex shrink.

        # iter2[0] - modifies (2,3,5)
        # M = (1 8/9, 2 8/9, 4 4/9)
        np.testing.assert_almost_equal(iter_2[0]['v1__FREE__'], 1 + 7./9.)
        np.testing.assert_almost_equal(iter_2[0]['v2__FREE__'], 2. + 7./9.)

        res = algorithms.Result(iter_2[0], self.data1s, iter_2[0].name)
        res.score = 5.9
        p3_1 = sim.got_result(res)
        # Should move half way back to the centroid
        np.testing.assert_almost_equal(p3_1[0]['v1__FREE__'], 1 + 15./18.)
        np.testing.assert_almost_equal(p3_1[0]['v2__FREE__'], 2. + 15./18.)
        res = algorithms.Result(p3_1[0], self.data1s, p3_1[0].name)
        res.score = 10.
        nothing = sim.got_result(res)

        # iter2[1] modifies (1, 3 2/3, 4 2/3)
        # M = (2 2/9, 2 2/3, 4 5/9)
        res = algorithms.Result(iter_2[1], self.data1s, iter_2[1].name)
        res.score = 5.9
        p3_2 = sim.got_result(res)
        # Worse than original, should move to halfway to the centroid.
        np.testing.assert_almost_equal(p3_2[0]['v1__FREE__'], 1. + 11. / 18.)
        np.testing.assert_almost_equal(p3_2[0]['v2__FREE__'], 3. + 1. / 6.)
        res = algorithms.Result(p3_2[0], self.data1s, p3_2[0].name)
        res.score = 10.
        final_ps = sim.got_result(res)

        # After replacement, we have
        # ((2 2/3, 2, 4 2/3), (2,3,4), (1, 3 2/3, 4 2/3), (1 7/9, 2 7/9, 3 8/9))

        # Iteration was unproductive, so expect to shrink, should have gotten 3 new psets to complete that shrink
        assert len(final_ps) == 3
        np.testing.assert_almost_equal(final_ps[0]['v1__FREE__'], 7. / 3.)
        np.testing.assert_almost_equal(final_ps[1]['v2__FREE__'], 17. / 6.)
        np.testing.assert_almost_equal(final_ps[2]['v3__FREE__'], 77. / 18.)

    def test_start_log(self):
        sim = algorithms.SimplexAlgorithm(deepcopy(self.logconfig))
        sim.variables.sort()  # Required for Python <= 3.5 to be sure we are checking the correct indices in the simplex
        first = sim.start_run()
        assert len(first) == 4
        np.testing.assert_almost_equal(first[0]['v1__FREE__'], 10.**2.)
        np.testing.assert_almost_equal(first[3]['v3__FREE__'], 10.**5.)

    def test_updates_log(self):
        """ The above test should also work identically in log space"""
        sim = algorithms.SimplexAlgorithm(deepcopy(self.logconfig))
        sim.variables.sort()  # Required for Python <= 3.5 to be sure we are checking the correct indices in the simplex
        first = sim.start_run()
        next_params = []
        for p, score in zip(first, [5., 7., 8., 6.]):
            res = algorithms.Result(p, self.data1s, p.name)
            res.score = score
            next_params += sim.got_result(res)

        assert len(next_params) == 2
        # First point should be the reflection of the worst PSet, where v2 += 1
        # M = (2 1/3, 3, 4 1/3)
        np.testing.assert_almost_equal(next_params[0]['v1__FREE__'], 10**(8. / 3.))
        # Second point should be reflection of v1
        # M = (2, 3 1/3, 4 1/3)
        np.testing.assert_almost_equal(next_params[1]['v1__FREE__'], 10**1.)

        res = algorithms.Result(next_params[1], self.data1s, next_params[1].name)
        res.score = 5.5
        # Case 2 - take it, do nothing else.
        p2_1 = sim.got_result(res)
        assert p2_1 == []
        res = algorithms.Result(next_params[0], self.data1s, next_params[0].name)
        res.score = 4.5
        # Case 1 - pick an expansion point
        p2_2 = sim.got_result(res)
        assert len(p2_2) == 1
        # print(sim.centroids)
        # print(p2_2)
        np.testing.assert_almost_equal(p2_2[0]['v1__FREE__'], 10**3.)
        np.testing.assert_almost_equal(p2_2[0]['v2__FREE__'], 10**1.)

        res = algorithms.Result(p2_2[0], self.data1s, p2_2[0].name)
        res.score = 4.75
        iter_2 = sim.got_result(res)
        assert len(iter_2) == 2
        assert sim.simplex == [(4.5, next_params[0]), (5., first[0]), (5.5, next_params[1]), (6., first[3])]
        # ((2 2/3, 2, 4 2/3), (2,3,4), (1, 3 2/3, 4 2/3), (2,3,5))

        # Next iteration. This time, we'll give non-improvements, forcing a simplex shrink.

        # iter2[0] - modifies (2,3,5)
        # M = (1 8/9, 2 8/9, 4 4/9)
        np.testing.assert_almost_equal(iter_2[0]['v1__FREE__'], 10**(1 + 7. / 9.))
        np.testing.assert_almost_equal(iter_2[0]['v2__FREE__'], 10**(2. + 7. / 9.))

        res = algorithms.Result(iter_2[0], self.data1s, iter_2[0].name)
        res.score = 5.9
        p3_1 = sim.got_result(res)
        # Should move half way back to the centroid
        np.testing.assert_almost_equal(p3_1[0]['v1__FREE__'], 10**(1 + 15. / 18.))
        np.testing.assert_almost_equal(p3_1[0]['v2__FREE__'], 10**(2. + 15. / 18.))
        res = algorithms.Result(p3_1[0], self.data1s, p3_1[0].name)
        res.score = 10.
        nothing = sim.got_result(res)

        # iter2[1] modifies (1, 3 2/3, 4 2/3)
        # M = (2 2/9, 2 2/3, 4 5/9)
        res = algorithms.Result(iter_2[1], self.data1s, iter_2[1].name)
        res.score = 5.9
        p3_2 = sim.got_result(res)
        # Worse than original, should move to halfway to the centroid.
        np.testing.assert_almost_equal(p3_2[0]['v1__FREE__'], 10**(1. + 11. / 18.))
        np.testing.assert_almost_equal(p3_2[0]['v2__FREE__'], 10**(3. + 1. / 6.))
        res = algorithms.Result(p3_2[0], self.data1s, p3_2[0].name)
        res.score = 10.
        final_ps = sim.got_result(res)

        # After replacement, we have
        # ((2 2/3, 2, 4 2/3), (2,3,4), (1, 3 2/3, 4 2/3), (1 7/9, 2 7/9, 3 8/9))

        # Iteration was unproductive, so expect to shrink, should have gotten 3 new psets to complete that shrink
        assert len(final_ps) == 3
        np.testing.assert_almost_equal(final_ps[0]['v1__FREE__'], 10**(7. / 3.))
        np.testing.assert_almost_equal(final_ps[1]['v2__FREE__'], 10**(17. / 6.))
        np.testing.assert_almost_equal(final_ps[2]['v3__FREE__'], 10**(77. / 18.))
