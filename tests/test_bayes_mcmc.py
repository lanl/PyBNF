from .context import data, algorithms, pset, objective, config
import os
import shutil


class TestBayes:
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

        cls.variables = ['v1', 'v2', 'v3']

        cls.chi_sq = objective.ChiSquareObjective()

        cls.params = pset.PSet({'v1': 3.14, 'v2': 1.0, 'v3': 0.1})
        cls.params2 = pset.PSet({'v1': 4.14, 'v2': 10.0, 'v3': 1.0})

        os.makedirs('noseoutput1/Results', exist_ok=True)
        os.makedirs('noseoutput2/Results', exist_ok=True)

        # Note mutation_rate is set to 1.0 because for tests with few params, with a lower mutation_rate might randomly
        # create a duplicate parameter set, causing the "not in individuals" tests to fail.
        cls.config = config.Configuration({
            'population_size': 20, 'max_iterations': 20, 'step_size': 0.2, 'output_hist_every': 10, 'sample_every': 2,
            'burn_in': 3, 'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': 'noseoutput1/',
            ('lognormrandom_var', 'v1'): [0., 0.5], ('lognormrandom_var', 'v2'): [0., 0.5], ('random_var', 'v3'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
            'bng_command': 'For this test you don''t need this.'})

        cls.config_box = config.Configuration({
            'population_size': 20, 'max_iterations': 20, 'step_size': 0.2, 'output_hist_every': 10, 'sample_every': 2,
            'burn_in': 3, 'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': 'noseoutput1/',
            ('random_var', 'v1'): [0, 10], ('random_var', 'v2'): [0, 10],
            ('random_var', 'v3'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
            'bng_command': 'For this test you don''t need this.'})

        cls.config_normal = config.Configuration({
            'population_size': 20, 'max_iterations': 20, 'step_size': 0.2, 'output_hist_every': 10, 'sample_every': 2,
            'burn_in': 3, 'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': 'noseoutput2/',
            ('lognormrandom_var', 'v1'): [0., 0.5], ('lognormrandom_var', 'v2'): [0., 0.5],
            ('lognormrandom_var', 'v3'): [0., 0.5],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
            'bng_command': 'For this test you don''t need this.'})

    @classmethod
    def teardown_class(cls):
        shutil.rmtree('noseoutput1')

    def test_start(self):
        ba = algorithms.BayesAlgorithm(self.config)
        start_params = ba.start_run()
        assert len(start_params) == 20
        assert ba.prior['v1'] == ('n', 0., 0.5)
        assert ba.prior['v3'] == ('b', 0, 10)

    def test_updates_box(self):
        # In this test, the variables have box constraints, so the prior contribution should be constant.
        # We test the decisions to replace / not replace, which should be fairly deterministic

        ba = algorithms.BayesAlgorithm(self.config_box)
        start_params = ba.start_run()
        next_params = []
        for p in start_params:
            res = algorithms.Result(p, self.data1s, [''], p.name)
            res.score = 42.
            next_params += ba.got_result(res)

        params_3 = []
        for p in next_params:
            res = algorithms.Result(p, self.data1s, [''], p.name)
            res.score = 42.693  # Should accept with probability 0.5
            params_3 += ba.got_result(res)

        replaced = 0
        kept = 0
        for pp in ba.current_pset:
            if pp in start_params:
                kept += 1
            if pp in next_params:
                replaced += 1
        print(kept, replaced)
        # If accept is correctly 50%, these should pass with probability 1 - 4e-05
        assert 2 <= replaced <= 18
        assert 2 <= kept <= 18
        assert kept + replaced == 20

        params_4 = []
        for p in params_3:
            res = algorithms.Result(p, self.data1s, [''], p.name)
            res.score = 41.9999  # Should accept with probability 1
            params_4 += ba.got_result(res)
        for pp in ba.current_pset:
            assert pp in params_3

    def test_updates_normal(self):
        # In this test, variables have lognormal priors. This makes the overall fitness of psets random, depending on
        # prior
        # But, iteration and output count is deterministic, and we test that here.
        ba = algorithms.BayesAlgorithm(self.config_normal)

        # Run 10 iterations
        start_params = ba.start_run()
        curr_params = start_params
        for i in range(10):
            next_params = []
            for p in curr_params:
                res = algorithms.Result(p, self.data1s, [''], p.name)
                res.score = 42.
                next_params += ba.got_result(res)
            curr_params = next_params

