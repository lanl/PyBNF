from .context import algorithms, data, objective
import copy


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

        cls.ps = algorithms.ParticleSwarm(cls.d1e, cls.chi_sq, {})

    def test_start(self):
        ps = copy.deepcopy(self.ps)  # Use a fresh copy of the algorithm for each test.
        start_params = ps.start_run()
        assert len(start_params) == 15

    def test_updates(self):
        ps = copy.deepcopy(self.ps)  # Use a fresh copy of the algorithm for each test.
        start_params = ps.start_run()
        next_params = []
        for p in start_params:
            next_params += ps.got_result(p, self.d2s)

        assert ps.global_best[0] in start_params

        ps.got_result(next_params[7], self.d1s) # better than the previous ones
        assert ps.global_best[0] == next_params[7]

        # Exactly 1 individual particle should have its best as that global best, the rest should be one of start_params
        count = 0
        for i in range(15):
            if ps.bests[i][0] == next_params[7]:
                count += 1
            else:
                assert ps.bests[i][0] in start_params
        assert count == 1


    def test_example_run(self):
        # Requires complete rewrite with new organization scheme.
        pass