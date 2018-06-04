from .context import data, algorithms, pset, objective, config
import os
import shutil
import numpy as np


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

        os.makedirs('noseoutput1/Results', exist_ok=True)
        os.makedirs('noseoutput2/Results', exist_ok=True)

        cls.p0 = pset.FreeParameter('v1__FREE', 'lognormal_var', 1., 0.5, 4.14)
        cls.p1 = pset.FreeParameter('v2__FREE', 'lognormal_var', 1., 0.5, 10.0)
        cls.p2 = pset.FreeParameter('v3__FREE', 'normal_var', 50, 3, 1.0)
        cls.pset = pset.PSet([cls.p0, cls.p1, cls.p2])

        # Note mutation_rate is set to 1.0 because for tests with few params, with a lower mutation_rate might randomly
        # create a duplicate parameter set, causing the "not in individuals" tests to fail.
        cls.config = config.Configuration({
            'population_size': 20, 'max_iterations': 20, 'step_size': 0.2, 'output_hist_every': 5, 'sample_every': 2,
            'burn_in': 3, 'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': 'noseoutput1/',
            ('lognormal_var', 'v1__FREE'): [0., 0.5], ('loguniform_var', 'v2__FREE'): [1., 10.], ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'fit_type': 'bmc'})

        cls.config_box = config.Configuration({
            'population_size': 20, 'max_iterations': 20, 'step_size': 0.2, 'output_hist_every': 5, 'sample_every': 2,
            'burn_in': 3, 'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': 'noseoutput1/',
            ('uniform_var', 'v1__FREE'): [0, 10], ('uniform_var', 'v2__FREE'): [0, 10],
            ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'fit_type': 'bmc'})

        cls.config_normal = config.Configuration({
            'population_size': 20, 'max_iterations': 20, 'step_size': 0.2, 'output_hist_every': 5, 'sample_every': 2,
            'burn_in': 3, 'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': 'noseoutput2/',
            ('lognormal_var', 'v1__FREE'): [0., 0.5], ('lognormal_var', 'v2__FREE'): [0., 0.5],
            ('lognormal_var', 'v3__FREE'): [0., 0.5],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'fit_type': 'bmc'})

        cls.config_replica = config.Configuration({
            'population_size': 4, 'max_iterations': 20, 'step_size': 0.2, 'output_hist_every': 5, 'sample_every': 2,
            'burn_in': 3, 'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': 'noseoutput1/',
            ('lognormal_var', 'v1__FREE'): [1., 0.5], ('lognormal_var', 'v2__FREE'): [1., 0.5], ('normal_var', 'v3__FREE'): [50, 3],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'exchange_every': 5, 'beta': [1., 0.9, 0.8, 0.7], 'fit_type': 'pt'})

        cls.config_replica_multi = config.Configuration({
            'population_size': 8, 'max_iterations': 20, 'step_size': 0.2, 'output_hist_every': 5, 'sample_every': 2,
            'burn_in': 3, 'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': 'noseoutput1/',
            ('lognormal_var', 'v1__FREE'): [1., 0.5], ('lognormal_var', 'v2__FREE'): [1., 0.5], ('normal_var', 'v3__FREE'): [50, 3],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'exchange_every': 5, 'beta': [1., 0.9, 0.8, 0.7],
            'fit_type': 'pt', 'reps_per_beta': 2})

    @classmethod
    def teardown_class(cls):
        shutil.rmtree('noseoutput1')
        # shutil.rmtree('noseoutput2')

    def test_start(self):
        ba = algorithms.BasicBayesMCMCAlgorithm(self.config)
        start_params = ba.start_run()
        assert len(start_params) == 20
        assert ba.prior['v1__FREE'] == ('log', 'n', 0., 0.5)
        assert ba.prior['v2__FREE'] == ('log', 'b', 0., 1.)
        assert ba.prior['v3__FREE'] == ('reg', 'b', 0, 10)

    def test_updates_box(self):
        # In this test, the variables have box constraints, so the prior contribution should be constant.
        # We test the decisions to replace / not replace, which should be fairly deterministic

        ba = algorithms.BasicBayesMCMCAlgorithm(self.config_box)
        start_params = ba.start_run()
        next_params = []
        for p in start_params:
            res = algorithms.Result(p, self.data1s, p.name)
            res.score = 42.
            next_params += ba.got_result(res)

        params_3 = []
        for p in next_params:
            res = algorithms.Result(p, self.data1s, p.name)
            res.score = 42.693  # Should accept with probability 0.5
            params_3 += ba.got_result(res)

        replaced = 0
        kept = 0
        for pp in ba.current_pset:
            if pp in start_params:
                kept += 1
            if pp in next_params:
                replaced += 1
        # print(kept, replaced)
        # If accept is correctly 50%, these should pass with probability 1 - 4e-05
        assert 2 <= replaced <= 18
        assert 2 <= kept <= 18
        assert kept + replaced == 20

        params_4 = []
        for p in params_3:
            res = algorithms.Result(p, self.data1s, p.name)
            res.score = 41.9999  # Should accept with probability 1
            params_4 += ba.got_result(res)
        for pp in ba.current_pset:
            assert pp in params_3

    def test_updates_normal(self):
        # In this test, variables have lognormal priors. This makes the overall fitness of psets random, depending on
        # prior
        # But, iteration and output count is deterministic, and we test that here.
        ba = algorithms.BasicBayesMCMCAlgorithm(self.config_normal)

        # Run 10 iterations
        start_params = ba.start_run()
        curr_params = start_params
        for i in range(10):
            next_params = []
            for p in curr_params:
                res = algorithms.Result(p, self.data1s, p.name)
                res.score = 42.
                next_params += ba.got_result(res)
            curr_params = next_params

        # Check the files came out looking reasonable
        a = np.genfromtxt('noseoutput2/Results/Histograms/v1__FREE_10.txt')
        assert a.shape == (10, 3)
        assert sum(a[:, 2]) == 80  # 20 saves on iters 4, 6, 8, and 10

        s = np.genfromtxt('noseoutput2/Results/samples.txt', usecols=(1, 2, 3, 4))
        assert s.shape[0] == 80
        # Don't know what the Names in s should be; depends what PSets got randomly kept in the population.

        with open('noseoutput2/Results/credible68_10.txt') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    assert line == '# param\tlower_bound\tupper_bound\n'
                else:
                    parts = line.split('\t')
                    assert parts[0] in [v.name for v in ba.variables]
                    assert float(parts[1]) < float(parts[2])

    def test_replica_exchange_run(self):
        ba = algorithms.BasicBayesMCMCAlgorithm(self.config_replica)
        start_params = ba.start_run()
        assert len(start_params) == 4
        for chain in range(4):
            ps = start_params[chain]
            # Send in the first 5 results, which should get this chain to the synchronization point.
            for i in range(5):
                res = algorithms.Result(ps, self.data1s, ps.name)
                res.score = 42.
                nextlist = ba.got_result(res)
                if i<4:
                    assert len(nextlist) == 1
                    ps = nextlist[0]
                else:
                    if chain < 3:
                        assert len(nextlist) == 0
                        assert ba.wait_for_sync[chain] is True
                    else:
                        assert len(nextlist) == 4
                        assert not np.any(ba.wait_for_sync)

    def test_replica_exchange_function(self):
        count = 0
        for iters in range(20):
            ba = algorithms.BasicBayesMCMCAlgorithm(self.config_replica)
            psets = []
            for i in range(4):
                p = self.pset
                p.name = 'iter0run%i' % i
                psets.append(p)
            ba.current_pset = psets
            ba.ln_current_P = [-5000., -1., -2., 3.]
            ba.replica_exchange()
            assert ba.ln_current_P[0] == -5000.
            assert ba.ln_current_P[1] == -2.
            print(ba.ln_current_P)
            if ba.ln_current_P[2] == 3:
                count+=1
        assert 0 < count < 20

    def test_replica_exchange_multi(self):
        count = 0
        for iters in range(20):
            ba = algorithms.BasicBayesMCMCAlgorithm(self.config_replica_multi)
            assert ba.betas == [0.7, 0.8, 0.9, 1., 0.7, 0.8, 0.9, 1.]
            assert ba.should_sample(3)
            assert not ba.should_sample(4)
            psets = []
            for i in range(8):
                p = self.pset
                p.name = 'iter0run%i' % i
                psets.append(p)
            ba.current_pset = psets
            ba.ln_current_P = [-1., -5000., -2., 3., -1., -42., -2., 3.]
            ba.replica_exchange()
            assert ((ba.ln_current_P[0] == -5000. and ba.ln_current_P[4] == -42.) or
                    (ba.ln_current_P[4] == -5000. and ba.ln_current_P[0] == -42.))
            if ba.ln_current_P[0] == -5000. and ba.ln_current_P[4] == -42:
                count += 1
        assert 0 < count < 20
