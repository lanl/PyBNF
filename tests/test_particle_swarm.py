from .context import data, algorithms, pset, objective, config, parse
import numpy as np
import numpy.testing as npt
from os import environ
from shutil import rmtree


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

        cls.variables = ['v1', 'v2', 'v3']

        cls.chi_sq = objective.ChiSquareObjective()

        cls.config = config.Configuration({'population_size': 15, 'max_iterations': 20, 'cognitive': 1.5, 'social': 1.5,
                      ('random_var', 'v1'): [0, 10], ('random_var', 'v2'): [0, 10], ('random_var', 'v3'): [0, 10],
                      'models': {'bngl_files/parabola.bngl'}, 'exp_data':{'bngl_files/par1.exp'},
                      'bngl_files/parabola.bngl':['bngl_files/par1.exp'],
                      'bng_command': 'For this test you don''t need this.'})

        cls.config2 = config.Configuration({'population_size': 15, 'max_iterations': 20, 'cognitive': 1.5, 'social': 1.5,
                           ('static_list_var', 'v1'): [17., 42., 3.14], ('loguniform_var', 'v2'): [0.01, 1e5],
                           ('lognormrandom_var', 'v3'): [0, 1],
                           'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
                           'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
                           'bng_command': 'For this test you don''t need this.'})

        cls.config_path = 'bngl_files/parabola.conf'

    @classmethod
    def teardown_class(cls):
        for n in range(1, 312):
            try:
                rmtree('sim_'+str(n))
            except FileNotFoundError:
                # Exactly how many sims were done depends on random run timing, so some might be missing.
                pass

    def test_random_pset(self):
        ps = algorithms.ParticleSwarm(self.config2)
        params = ps.random_pset()
        # Necessary but not sufficient check that this is working correctly.
        assert params['v1'] in [17., 42., 3.14]
        assert 0.01 < params['v2'] < 1e5
        assert 1e-4 < params['v3'] < 1e4

    def test_start(self):
        ps = algorithms.ParticleSwarm(self.config)
        start_params = ps.start_run()
        assert len(start_params) == 15

    def test_updates(self):
        ps = algorithms.ParticleSwarm(self.config)
        start_params = ps.start_run()
        next_params = []
        for p in start_params:
            new_result = algorithms.Result(p, self.d2s, '')
            new_result.score = ps.objective.evaluate(self.d2s, self.d1e)
            next_params += ps.got_result(new_result)

        assert ps.global_best[0] in start_params

        new_result = algorithms.Result(next_params[7], self.d1s, '')
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

    def test_full(self):
        conf_dict = parse.load_config(self.config_path)
        myconfig = config.Configuration(conf_dict)
        myconfig.config['bng_command'] = environ['BNGPATH'] + '/BNG2.pl'
        ps = algorithms.ParticleSwarm(myconfig)
        ps.run()
        # print(ps.global_best)
        best_fit = ps.global_best[0]

        # The data is most sensitive to the x^2 coefficent, so this gets fit the best.
        # Here's a reasonable test that the fitting went okay.
        assert abs(best_fit['v1__FREE__'] - 0.5) < 0.3

