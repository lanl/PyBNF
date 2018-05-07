from .context import data, algorithms, pset, objective, config, parse
import numpy.testing as npt
from os import mkdir, path
from shutil import rmtree
from copy import deepcopy


class TestParticleSwarm:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.data1e = [
            '# time    v1_result    v2_result    v3_result  v1_result_SD  v2_result_SD  v3_result_SD\n',
            ' 0 3   4   5   0.1   0.2   0.3\n',
            ' 1 2   3   6   0.1   0.1   0.1\n',
            ' 2 4   2   10  0.3   0.1   1.0\n'
        ]

        cls.d1e = data.Data()
        cls.d1e.data = cls.d1e._read_file_lines(cls.data1e, '\s+')

        cls.data1s = [
            '# time    v1_result    v2_result    v3_result\n',
            ' 1 2.1   3.1   6.1\n',
        ]
        cls.d1s = data.Data()
        cls.d1s.data = cls.d1s._read_file_lines(cls.data1s, '\s+')

        cls.data2s = [
            '# time    v1_result    v2_result    v3_result\n',
            ' 1 2.2   3.2   6.2\n',
        ]
        cls.d2s = data.Data()
        cls.d2s.data = cls.d2s._read_file_lines(cls.data2s, '\s+')

        cls.variables = ['v1__FREE', 'v2__FREE', 'v3__FREE']

        cls.chi_sq = objective.ChiSquareObjective()

        cls.p0 = pset.FreeParameter('v1__FREE', 'uniform_var', 0, 10, 3.14)
        cls.p1 = pset.FreeParameter('v2__FREE', 'uniform_var', 0, 10, 1.0)
        cls.p2 = pset.FreeParameter('v3__FREE', 'uniform_var', 0, 10, 0.1)

        cls.params = pset.PSet([cls.p0, cls.p1, cls.p2])

        cls.config = config.Configuration({'population_size': 15, 'max_iterations': 20, 'cognitive': 1.5, 'social': 1.5,
                      ('uniform_var', 'v1__FREE'): [0, 10], ('uniform_var', 'v2__FREE'): [0, 10], ('uniform_var', 'v3__FREE'): [0, 10],
                      'models': {'bngl_files/parabola.bngl'}, 'exp_data':{'bngl_files/par1.exp'},
                      'bngl_files/parabola.bngl':['bngl_files/par1.exp'],
                      'fit_type': 'pso', 'output_dir': 'test_pso'})

        cls.config2 = config.Configuration({'population_size': 15, 'max_iterations': 20, 'cognitive': 1.5, 'social': 1.5,
                           ('uniform_var', 'v1__FREE'): [0, 10], ('loguniform_var', 'v2__FREE'): [0.01, 1e5],
                           ('lognormal_var', 'v3__FREE'): [0, 1],
                           'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
                           'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
                           'fit_type': 'pso', 'output_dir': 'test_pso2'})

        cls.config_path = 'bngl_files/parabola.conf'

        cls.lh_config = config.Configuration(
            {'population_size': 10, 'max_iterations': 20, 'cognitive': 1.5, 'social': 1.5,
            ('uniform_var', 'v1__FREE'): [0, 10], ('uniform_var', 'v2__FREE'): [0, 10], ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'output_dir': 'test_pso_lh',
            'initialization': 'lh', 'fit_type': 'pso'})

    @classmethod
    def teardown_class(cls):
        if path.isdir('test_pso_lh'):
            rmtree('test_pso_lh')
        if path.isdir('test_pso2'):
            rmtree('test_pso2')
        if path.isdir('test_pso'):
            rmtree('test_pso')

    def test_random_pset(self):
        ps = algorithms.ParticleSwarm(deepcopy(self.config2))
        params = ps.random_pset()
        assert 0 <= params['v1__FREE'] <= 10
        assert 0.01 < params['v2__FREE'] < 1e5
        assert 1e-4 < params['v3__FREE'] < 1e4

    def test_start(self):
        ps = algorithms.ParticleSwarm(self.config)
        start_params = ps.start_run()
        assert len(start_params) == 15

    def test_updates(self):
        ps = algorithms.ParticleSwarm(self.config)
        start_params = ps.start_run()
        next_params = []
        for p in start_params:
            new_result = algorithms.Result(p, self.d2s, 'sim_1')
            new_result.score = ps.objective.evaluate(self.d2s, self.d1e)
            next_params += ps.got_result(new_result)

        assert ps.global_best[0] in start_params

        new_result = algorithms.Result(next_params[7], self.d1s, 'sim_1')
        new_result.score = ps.objective.evaluate(self.d1s, self.d1e)
        ps.got_result(new_result)  # better than the previous ones
        assert ps.global_best[0] == next_params[7]

        # Exactly 1 individual particle should have its best as that global best, the rest should be one of start_params
        count = 0
        for i in range(15):
            if ps.bests[i][0] == next_params[7]:
                count += 1
            else:
                assert ps.bests[i][0] in start_params
        assert count == 1

    def test_latin_hypercube(self):
        ps = algorithms.ParticleSwarm(self.lh_config)
        ps.start_run()
        for i in range(10):
            # Latin hypercube should distribute starting values evenly (one in each bin) in each dimension.
            assert len([x for x in ps.swarm if i < x[0]['v1__FREE'] < i+1]) == 1


