from .context import data, algorithms, pset, objective, config

import shutil


class TestDiffEvolution:
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

        cls.params = pset.PSet({'v1': 3.14, 'v2': 1.0, 'v3': 0.1})
        cls.params2 = pset.PSet({'v1': 4.14, 'v2': 10.0, 'v3': 1.0})

        # Note mutation_rate is set to 1.0 because for tests with few params, with a lower mutation_rate might randomly
        # create a duplicate parameter set, causing the "not in individuals" tests to fail.
        cls.config = config.Configuration({
            'population_size': 20, 'max_iterations': 20, 'islands': 2, 'migrate_every': 3, 'num_to_migrate': 2,
            'mutation_rate': 1.0, 'fit_type': 'de',
            ('random_var', 'v1'): [0, 10], ('random_var', 'v2'): [0, 10], ('random_var', 'v3'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
            'output_dir': 'test_init'})

    @classmethod
    def teardown_class(cls):
        shutil.rmtree('test_init')

    def test_start(self):
        de = algorithms.DifferentialEvolution(self.config)
        assert de.num_per_island == 10
        start_params = de.start_run()
        assert len(start_params) == 20
        assert len(de.individuals) == 2
        assert len(de.individuals[0]) == 10
        assert de.waiting_count == [10, 10]

    def test_updates(self):
        de = algorithms.DifferentialEvolution(self.config)
        start_params = de.start_run()

        for i in range(9):
            res = algorithms.Result(start_params[i], self.data1s, [''], start_params[i].name)
            res.score = 42.
            torun = de.got_result(res)
            assert torun == []
        # Finish island 1 iter 0, should get some new params.
        res = algorithms.Result(start_params[9], self.data1s, [''], start_params[9].name)
        res.score = 42.
        torun = de.got_result(res)
        assert len(torun) == 10
        next_params = torun
        assert de.iter_num == [1, 0]
        for i in range(10, 20):
            res = algorithms.Result(start_params[i], self.data1s, [''], start_params[i].name)
            res.score = 150.
            torun = de.got_result(res)
            next_params += torun
        # End of iteration 0
        assert de.iter_num == [1, 1]

        params_gen2 = []
        for i in range(20):
            res = algorithms.Result(next_params[i], self.data1s, [''], next_params[i].name)
            res.score = max(1., i ** 2)
            if i < 10:
                assert de.island_map[next_params[i]] == (0, i)
            else:
                assert de.island_map[next_params[i]] == (1, i-10)
            torun = de.got_result(res)
            # Replace if i**2 is better than previous value
            if i <= 6:
                assert next_params[i] == de.individuals[0][i]
            elif 7 <= i <= 9:
                assert start_params[i] == de.individuals[0][i]
            elif 10 <= i <= 12:
                assert next_params[i] == de.individuals[1][i-10]
            elif 12 < i:
                assert start_params[i] == de.individuals[1][i-10]
            if i == 9 or i == 19:
                assert len(torun) == 10
            else:
                assert len(torun) == 0
            params_gen2 += torun
        # End of iteration 1
        assert de.iter_num == [2, 2]

        # After iteration 2, migration will trigger
        params_gen3 = []
        for i in range(10):
            res = algorithms.Result(params_gen2[i], self.data1s, [''], params_gen2[i].name)
            res.score = 9999.
            torun = de.got_result(res)
            params_gen3 += torun
        assert de.migration_ready == [1, 0]
        assert de.migration_done == [0, 0]
        assert len(de.migration_indices[1]) == 2
        assert len(de.migration_perms[1]) == 2
        assert len(de.migration_transit[1][0]) == 2
        assert len(de.migration_transit[1][1]) == 0

        for i in range(10, 20):
            res = algorithms.Result(params_gen2[i], self.data1s, [''], params_gen2[i].name)
            res.score = 9999.
            torun = de.got_result(res)
            params_gen3 += torun

        assert de.migration_ready == [1, 1]
        assert de.migration_done == [0, 1]
