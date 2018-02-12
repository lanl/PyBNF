import math
import numpy as np
import numpy.testing as npt
from .context import data, printing
from nose.tools import raises

# Doesn't print warnings when dividing by zero
np.seterr(invalid='ignore', divide='ignore')


class TestData:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.str0 = "inf"
        cls.str1 = "-iNf"
        cls.str2 = "NaN"
        cls.str3 = "6.022e-23"
        cls.str4 = "abc"

        cls.file0 = 'bngl_files/test.gdat'
        cls.data0 = [
            '#          time    fullyBoundAg        Bound2Ag        Bound1Ag          freeAg    fullyBoundAb\n',
            ' 0.00000000e+00  1.20000000e+01  8.00000000e+00  6.00000000e+00  0.00000000e+00  1.60000000e+01\n',
            ' 1.00000000e+00  1.20000000e+01  8.00000000e+00  6.00000000e+00  0.00000000e+00  1.60000000e+01\n'
        ]
        cls.d0 = data.Data()
        cls.d0.data = cls.d0._read_file_lines(cls.data0, '\s+')

        cls.data1 = [
            '# x    obs1    obs2    obs3\n',
            ' 0 3   4   5\n',
            ' 1 2   3   6\n',
            ' 2 4   2   10\n'
        ]
        cls.d1 = data.Data()
        cls.d1.data = cls.d1._read_file_lines(cls.data1, '\s+')

        cls.data1b = [
            '# x    obs1    obs2    obs3\n',
            ' 0 5   6   5\n',
            ' 1 7   1   6\n',
            ' 2 5   0   10\n'
        ]
        cls.d1b = data.Data()
        cls.d1b.data = cls.d1b._read_file_lines(cls.data1b, '\s+')

        cls.data2 = [
            '# x    obs1    obs2    obs3\n',
            ' 0 3   4   5\n'
            ' \n',
            ' 1 2   3   6\n',
            ' 2 4   2   10\n\n'
        ]

        cls.data3 = [
            '# x    obs1    obs2    obs3\n',
            ' 0 3   4   5\n',
            ' 1 23   6\n',
            ' 2 4   2   10\n'
        ]

    def test_number_reader(self):
        assert data.Data._to_number(self.str0) == math.inf
        assert not math.isfinite(data.Data._to_number(self.str1))
        assert math.isnan(data.Data._to_number(self.str2))
        assert data.Data._to_number(self.str3) == 6.022e-23

    def test_file_reader(self):
        loc_data = data.Data(file_name=self.file0)
        npt.assert_allclose(loc_data.data, self.d0.data)

    @raises(ValueError)
    def test_number_reader_failure(self):
        data.Data._to_number(self.str4)

    def test_read_file_lines(self):
        md = data.Data()
        loc_data = md._read_file_lines(self.data0, '\s+')
        md.data = loc_data
        assert md.cols['time'] == 0
        assert len(md.cols.keys()) == 6
        assert md.data.shape == (2, 6)
        assert md.data[0, 2] == 8

    def test_column_access(self):
        assert self.d0.cols['time'] == 0
        assert self.d1.cols['obs2'] == 2
        npt.assert_allclose(self.d0['time'], self.d0.data[:, 0])
        npt.assert_allclose(self.d1['obs1'], self.d1.data[:, 1])

    def test_row_access(self):
        assert np.array_equal(self.d0.get_row('time',0.), np.array([0.00000000e+00,1.20000000e+01,8.00000000e+00,6.00000000e+00,0.00000000e+00,1.60000000e+01]))
        assert np.array_equal(self.d1.get_row('obs3',10.), np.array([2., 4., 2., 10.]))
        assert self.d1.get_row('x',3.) == None

    @raises(KeyError)
    def test_column_access_failure(self):
        self.d0['thing']

    def test_dep_col(self):
        npt.assert_allclose(self.d0._dep_cols(0), np.array([[12., 8., 6., 0., 16.], [12., 8., 6., 0., 16.]]))
        npt.assert_allclose(self.d0._dep_cols(1), np.array([[0., 8., 6., 0., 16.], [1., 8., 6., 0., 16.]]))

    def test_ind_col(self):
        npt.assert_allclose(self.d0._ind_col(0), np.array([0., 1.]))
        npt.assert_allclose(self.d0._ind_col(5), np.array([16., 16.]))

    @raises(IndexError)
    def test_ind_col_failure(self):
        self.d0._dep_cols(6)

    def test_init_normalization(self):
        norm0 = self.d0.normalize_to_init()
        npt.assert_allclose(norm0, np.array([[0., 1., 1., 1., np.nan, 1.], [1., 1., 1., 1., np.nan, 1.]]))
        norm1 = self.d1.normalize_to_init()
        npt.assert_allclose(norm1, np.array(
            [[0., 1., 1., 1.], [1., 2. / 3., 3. / 4., 6. / 5.], [2., 4. / 3., 2. / 4., 10. / 5.]]))

    def test_max_normalization(self):
        norm0 = self.d0.normalize_to_peak()
        npt.assert_allclose(norm0, np.array([[0., 1., 1., 1., np.nan, 1.], [1., 1., 1., 1., np.nan, 1.]]))
        norm1 = self.d1.normalize_to_peak()
        npt.assert_allclose(norm1, np.array(
            [[0., 3. / 4., 4. / 4., 5. / 10.], [1., 2. / 4., 3. / 4., 6. / 10.], [2., 4. / 4., 2. / 4., 10. / 10.]]))

    def test_zero_normalization(self):
        norm0 = self.d0.normalize_to_zero()
        npt.assert_allclose(norm0, np.array([[0., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0]]))
        norm1 = self.d1.normalize_to_zero(bc=False)
        npt.assert_allclose(norm1, np.array(
            [[0., 0., 1. / np.std(self.d1.data[:, 2]), -2. / np.std(self.d1.data[:, 3])],
             [1., -1. / np.std(self.d1.data[:, 1]), 0., -1. / np.std(self.d1.data[:, 3])],
             [2., 1. / np.std(self.d1.data[:, 1]), -1. / np.std(self.d1.data[:, 2]), 3. / np.std(self.d1.data[:, 3])]]))
        norm2 = self.d1.normalize_to_zero()
        npt.assert_allclose(norm2, np.array(
            [[0., 0., 1. / np.std(self.d1.data[:, 2], ddof=1), -2. / np.std(self.d1.data[:, 3], ddof=1)],
             [1., -1. / np.std(self.d1.data[:, 1], ddof=1), 0., -1. / np.std(self.d1.data[:, 3], ddof=1)],
             [2., 1. / np.std(self.d1.data[:, 1], ddof=1), -1. / np.std(self.d1.data[:, 2], ddof=1),
              3. / np.std(self.d1.data[:, 3], ddof=1)]]))

    def test_average(self):
        ave = data.Data.average([self.d1, self.d1b])
        assert ave.data[1, 0] == 1
        assert ave['obs3'][0] == 5.
        npt.assert_almost_equal(ave['obs1'][2], 4.5)

    def test_whitespace(self):
        d = data.Data()
        d.data = d._read_file_lines(self.data2, '\s+')
        assert d['obs1'][1] == 2

    @raises(printing.PybnfError)
    def test_misformatted(self):
        d = data.Data()
        d._read_file_lines(self.data3, '\s+')
