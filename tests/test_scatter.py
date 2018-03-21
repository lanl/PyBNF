from .context import data, algorithms, pset, objective, config, parse, printing
from nose.tools import raises
from os import mkdir
from shutil import rmtree
from copy import deepcopy


class TestScatter:
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

        cls.variables = ['v1__FREE__', 'v2__FREE__', 'v3__FREE__']

        cls.chi_sq = objective.ChiSquareObjective()

        cls.params = pset.PSet({'v1__FREE__': 3.14, 'v2__FREE__': 1.0, 'v3__FREE__': 0.1})
        cls.params2 = pset.PSet({'v1__FREE__': 4.14, 'v2__FREE__': 10.0, 'v3__FREE__': 1.0})

        # Note mutation_rate is set to 1.0 because for tests with few params, with a lower mutation_rate might randomly
        # create a duplicate parameter set, causing the "not in individuals" tests to fail.
        cls.config = config.Configuration({
            'population_size': 7, 'max_iterations': 20, 'fit_type': 'ss',
            ('uniform_var', 'v1__FREE__'): [0, 10], ('uniform_var', 'v2__FREE__'): [0, 10], ('uniform_var', 'v3__FREE__'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp']})

        cls.config_path = 'bngl_files/parabola.conf'
        mkdir('test_ss_output')
        mkdir('test_ss_output/Simulations')
        mkdir('test_ss_output/Results')
        mkdir('bnf_out')

    @classmethod
    def teardown_class(cls):
        rmtree('bnf_out')
        rmtree('test_ss_output')

    def test_start(self):
        ss = algorithms.ScatterSearch(deepcopy(self.config))
        ss.start_run()
        assert len(ss.refs) == 0
        assert len(ss.pending) == 30
        assert len(ss.reserve) == 20

    def test_updates(self):
        ss = algorithms.ScatterSearch(deepcopy(self.config))
        start_params = ss.start_run()
        ss.iteration = 1  # Avoid triggering output on iter 0.

        iter2run = []
        for i in range(30):
            res = algorithms.Result(start_params[i], self.data1s, start_params[i].name)
            res.score = 42.
            torun = ss.got_result(res)
            if i < 29:
                assert torun == []
            elif i == 29:
                assert len(torun) == 42  #pop_size*(pop_size-1)
                iter2run = torun
        assert len(ss.refs) == 7
        for p in ss.refs:
            assert len(ss.received[p[0]]) == 0

        # 2nd iteration
        i = 0
        out = ss.refs[3]
        notout = ss.refs[4]
        newref = None
        for pi in range(7):
            for hi in range(7):
                if pi == hi:
                    continue
                ps = iter2run[i]
                i += 1
                res = algorithms.Result(ps, self.data1s, ps.name)
                if pi == 3 and hi == 5:
                    res.score = 37.
                    newref = (ps, 37.)
                else:
                    res.score = 50.
                ss.got_result(res)

        assert out not in ss.refs
        assert notout in ss.refs
        assert newref in ss.refs
        assert ss.stuckcounter[notout[0]] == 1
        assert ss.stuckcounter[newref[0]] == 0

    def test_exp10(self):
        assert algorithms.exp10(2.) == 100.

    @raises(printing.PybnfError)
    def test_exp10_overflow(self):
        algorithms.exp10(100000.)
