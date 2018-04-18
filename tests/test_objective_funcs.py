from .context import data, objective, printing
import numpy.testing as npt
from nose.tools import raises


class TestObjectiveFunctions:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):

        cls.data1e = [
            '# x    obs1    obs2    obs3  obs1_SD  obs2_SD  obs3_SD\n',
            ' 0 3   4   5   0.1   0.2   0.3\n',
            ' 1 2   3   6   0.1   0.1   0.1\n',
            ' 2 4   2   10  0.3   0.1   1.0\n'
        ]
        cls.d1e = data.Data()
        cls.d1e.data = cls.d1e._read_file_lines(cls.data1e, '\s+')

        cls.data1s = [
            '# x    obs1    obs3\n',
            ' 0   3.1 5.1\n',
            ' 0.5 7   8\n',
            ' 1   2   6\n',
            ' 1.5 7   8\n',
            ' 2   4.2   10.2\n'
        ]
        cls.d1s = data.Data()
        cls.d1s.data = cls.d1s._read_file_lines(cls.data1s, '\s+')

        cls.data1s_nan = [
            '# x    obs1    obs3\n',
            ' 0   3.1 5.1\n',
            ' 0.5 7   8\n',
            ' 1   NaN   6\n',
            ' 1.5 7   8\n',
            ' 2   4.2   10.2\n'
        ]
        cls.d1s_nan = data.Data()
        cls.d1s_nan.data = cls.d1s_nan._read_file_lines(cls.data1s_nan, '\s+')

        cls.data1s_inf = [
            '# x    obs1    obs3\n',
            ' 0   3.1 5.1\n',
            ' 0.5 7   8\n',
            ' 1   2   Inf\n',
            ' 1.5 7   8\n',
            ' 2   4.2   10.2\n'
        ]
        cls.d1s_inf = data.Data()
        cls.d1s_inf.data = cls.d1s_inf._read_file_lines(cls.data1s_inf, '\s+')

        cls.chi_sq = objective.ChiSquareObjective()
        cls.sos = objective.SumOfSquaresObjective()
        cls.norm_sos = objective.NormSumOfSquaresObjective()
        cls.ave_norm_sos = objective.AveNormSumOfSquaresObjective()

    def test_chi_square(self):
        npt.assert_almost_equal(self.chi_sq.evaluate(self.d1s, self.d1e), 0.797777777777778)  # Value computed by hand

    @raises(printing.PybnfError)
    def test_chi_square_no_sd(self):
        self.chi_sq.evaluate(self.d1s, self.d1s)

    def test_sum_of_squares(self):
        npt.assert_almost_equal(self.sos.evaluate(self.d1s, self.d1e), 0.1)  # Value computed by hand

    def test_norm_sum_of_squares(self):
        npt.assert_almost_equal(self.norm_sos.evaluate(self.d1s, self.d1e), 0.00441111111111)  # Value computed by hand

    def test_ave_norm_sum_of_squares(self):
        # Value computed by hand
        npt.assert_almost_equal(self.ave_norm_sos.evaluate(self.d1s, self.d1e), 0.00657963719, decimal=5)

    def test_obj_nan(self):
        assert self.chi_sq.evaluate(self.d1s_nan, self.d1e) is None
        assert self.sos.evaluate(self.d1s_nan, self.d1e) is None
        assert self.norm_sos.evaluate(self.d1s_nan, self.d1e) is None
        assert self.ave_norm_sos.evaluate(self.d1s_nan, self.d1e) is None

    def test_obj_inf(self):
        assert self.chi_sq.evaluate(self.d1s_inf, self.d1e) is None
        assert self.sos.evaluate(self.d1s_inf, self.d1e) is None
        assert self.norm_sos.evaluate(self.d1s_inf, self.d1e) is None
        assert self.ave_norm_sos.evaluate(self.d1s_inf, self.d1e) is None
