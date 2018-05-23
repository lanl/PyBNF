from .context import algorithms, objective, data, config

import os
import re
import shutil


class TestDream:

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

        cls.variables = ['v1__FREE', 'v2__FREE', 'v3__FREE']

        cls.chi_sq = objective.ChiSquareObjective()

        os.makedirs('noseoutput1/Results', exist_ok=True)
        os.makedirs('noseoutput2/Results', exist_ok=True)

        cls.config = config.Configuration({
            'population_size': 20, 'max_iterations': 20, 'step_size': 0.2, 'output_hist_every': 10, 'sample_every': 2,
            'burn_in': 3, 'credible_intervals': [68, 95], 'num_bins': 10, 'output_dir': 'noseoutput1/',
            ('uniform_var', 'v1__FREE'): [0., 0.5], ('loguniform_var', 'v2__FREE'): [1., 10.], ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'fit_type': 'dream'})

    @classmethod
    def teardown_class(cls):
        shutil.rmtree('noseoutput1')
        shutil.rmtree('noseoutput2')

    def test_start(self):
        dream = algorithms.DreamAlgorithm(self.config)
        start_psets = dream.start_run()
        assert len(dream.variables) == 3
        assert sorted([v.name for v in dream.variables]) == ['v1__FREE', 'v2__FREE', 'v3__FREE']
        assert len(start_psets) == 20
        assert dream.prior['v1__FREE'] == ('reg', 'b', 0., 0.5)
        assert dream.prior['v2__FREE'] == ('log', 'b', 0., 1.)
        assert dream.prior['v3__FREE'] == ('reg', 'b', 0, 10)

    def test_update(self):
        dream = algorithms.DreamAlgorithm(self.config)
        start_psets = dream.start_run()
        for i, pset in enumerate(start_psets):
            res = algorithms.Result(pset, self.d1s, pset.name)
            res.score = 12
            if i == len(start_psets) - 1:
                next_gen = dream.got_result(res)
                assert len(next_gen) > 0  # unlikely that all updates are outside of the variable bounds
            else:
                empty = dream.got_result(res)
                assert empty == []

        for pset in next_gen:
            assert re.match('iter1run\d+', pset.name) is not None
