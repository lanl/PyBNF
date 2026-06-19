import math
import numpy as np
import numpy.testing as npt
from .context import data, algorithms, printing, raises
import copy

# Doesn't print warnings when dividing by zero
np.seterr(invalid='ignore', divide='ignore')


class TestData:
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
        cls.d0.data = cls.d0._read_file_lines(cls.data0, r'\s+')

        cls.data1 = [
            '# x    obs1    obs2    obs3\n',
            ' 0 3   4   5\n',
            ' 1 2   3   6\n',
            ' 2 4   2   10\n'
        ]
        cls.d1 = data.Data()
        cls.d1.data = cls.d1._read_file_lines(cls.data1, r'\s+')

        cls.data1b = [
            '# x    obs1    obs2    obs3\n',
            ' 0 5   6   5\n',
            ' 1 7   1   6\n',
            ' 2 5   0   10\n'
        ]
        cls.d1b = data.Data()
        cls.d1b.data = cls.d1b._read_file_lines(cls.data1b, r'\s+')

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

        cls.data1c = [
            '# x    obs1    obs2    obs3\n',
            ' # 0 5   6   5\n',
            ' 1 7   1   6\n',
            ' 2 5   0   10\n'
        ]
        cls.d1c = data.Data()
        cls.d1c.data = cls.d1c._read_file_lines(cls.data1c, r'\s+')

        cls.data1d = [
            '# x    obs1    obs2    obs3\n',
            ' 1 NaN   1   6\n',
            ' 2 5   Inf   10\n'
        ]
        cls.d1d = data.Data()
        cls.d1d.data = cls.d1d._read_file_lines(cls.data1d, r'\s+')

    def test_observer_pattern(self):
        d = data.Data()
        assert d.weights is None
        assert d.data is None
        d.data = np.arange(6).reshape(2,3)
        assert d.weights.shape == d.data.shape
        assert d.weights[0, 0] == 1
        assert d.weights[0, 1] == 1
        assert d.weights[0, 2] == 1
        assert d.weights[1, 0] == 1
        assert d.weights[1, 1] == 1
        assert d.weights[1, 2] == 1

    def test_valid_indices(self):
        vidcs = self.d1d._valid_indices()
        assert vidcs == [(0, 2), (0, 3), (1, 1), (1, 3)]

    def test_gen_bootstrap_weights(self):
        self.d1d.gen_bootstrap_weights(np.random.default_rng(0))
        print(self.d1d.weights)
        assert self.d1d.weights[0, 0] == 0
        assert self.d1d.weights[0, 1] == 0
        assert self.d1d.weights[0, 2] >= 0
        assert self.d1d.weights[0, 3] >= 0
        assert self.d1d.weights[1, 0] == 0
        assert self.d1d.weights[1, 1] >= 0
        assert self.d1d.weights[1, 2] == 0
        assert self.d1d.weights[1, 3] >= 0

    def test_comment_ignore(self):
        assert self.d1c.data.shape == (2, 4)
        assert self.d1c.data[0, 0] == 1

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
        loc_data = md._read_file_lines(self.data0, r'\s+')
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
        assert np.array_equal(self.d0.get_row('time', 0.), np.array(
            [0.00000000e+00, 1.20000000e+01, 8.00000000e+00, 6.00000000e+00, 0.00000000e+00, 1.60000000e+01]))
        assert np.array_equal(self.d1.get_row('obs3', 10.), np.array([2., 4., 2., 10.]))
        assert self.d1.get_row('x', 3.) is None

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
        d0 = copy.deepcopy(self.d0)
        d0.normalize_to_init()
        npt.assert_allclose(d0.data, np.array([[0., 1., 1., 1., np.nan, 1.], [1., 1., 1., 1., np.nan, 1.]]))
        d1 = copy.deepcopy(self.d1)
        d1.normalize_to_init()
        npt.assert_allclose(d1.data, np.array(
            [[0., 1., 1., 1.], [1., 2. / 3., 3. / 4., 6. / 5.], [2., 4. / 3., 2. / 4., 10. / 5.]]))
        d1b = copy.deepcopy(self.d1)
        d1b.normalize_to_init(cols=[1, 3])
        npt.assert_allclose(d1b.data, np.array(
            [[0., 1., 4., 1.], [1., 2. / 3., 3., 6. / 5.], [2., 4. / 3., 2., 10. / 5.]]))

    def test_subtract_baseline(self):
        d1 = copy.deepcopy(self.d1)
        d1._subtract_baseline()
        npt.assert_allclose(d1.data, np.array([[0, 0, 0, 0], [1, -1, -1, 1], [2, 1, -2, 5]]))

    def test_unit_scale(self):
        d1 = copy.deepcopy(self.d1)
        d1.normalize_to_unit_scale()
        npt.assert_allclose(d1.data, np.array([[0, 0, 0, 0], [1, -1.0, -0.5, 0.2], [2, 1.0, -1.0, 1.0]]))

    def test_max_normalization(self):
        d0 = copy.deepcopy(self.d0)
        d0.normalize_to_peak()
        npt.assert_allclose(d0.data, np.array([[0., 1., 1., 1., np.nan, 1.], [1., 1., 1., 1., np.nan, 1.]]))
        d1 = copy.deepcopy(self.d1)
        d1.normalize_to_peak()
        npt.assert_allclose(d1.data, np.array(
            [[0., 3. / 4., 4. / 4., 5. / 10.], [1., 2. / 4., 3. / 4., 6. / 10.], [2., 4. / 4., 2. / 4., 10. / 10.]]))
        d1b = copy.deepcopy(self.d1)
        d1b.normalize_to_peak(cols=[1, 3])
        npt.assert_allclose(d1b.data, np.array(
            [[0., 3. / 4., 4., 5. / 10.], [1., 2. / 4., 3., 6. / 10.], [2., 4. / 4., 2., 10. / 10.]]))

    def test_zero_normalization(self):
        d0 = copy.deepcopy(self.d0)
        d0.normalize_to_zero()
        npt.assert_allclose(d0.data, np.array([[0., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0]]))
        d1 = copy.deepcopy(self.d1)
        d1.normalize_to_zero(bc=False)
        npt.assert_allclose(d1.data, np.array(
            [[0., 0., 1. / np.std(self.d1.data[:, 2]), -2. / np.std(self.d1.data[:, 3])],
             [1., -1. / np.std(self.d1.data[:, 1]), 0., -1. / np.std(self.d1.data[:, 3])],
             [2., 1. / np.std(self.d1.data[:, 1]), -1. / np.std(self.d1.data[:, 2]), 3. / np.std(self.d1.data[:, 3])]]))
        d1b = copy.deepcopy(self.d1)
        d1b.normalize_to_zero(bc=False, cols=[1, 3])
        npt.assert_allclose(d1b.data, np.array(
            [[0., 0., 4., -2. / np.std(self.d1.data[:, 3])],
             [1., -1. / np.std(self.d1.data[:, 1]), 3., -1. / np.std(self.d1.data[:, 3])],
             [2., 1. / np.std(self.d1.data[:, 1]), 2., 3. / np.std(self.d1.data[:, 3])]]))
        d2 = copy.deepcopy(self.d1)
        d2.normalize_to_zero()
        npt.assert_allclose(d2.data, np.array(
            [[0., 0., 1. / np.std(self.d1.data[:, 2], ddof=1), -2. / np.std(self.d1.data[:, 3], ddof=1)],
             [1., -1. / np.std(self.d1.data[:, 1], ddof=1), 0., -1. / np.std(self.d1.data[:, 3], ddof=1)],
             [2., 1. / np.std(self.d1.data[:, 1], ddof=1), -1. / np.std(self.d1.data[:, 2], ddof=1),
              3. / np.std(self.d1.data[:, 3], ddof=1)]]))

    def test_average(self):
        ave = data.Data.average([self.d1, self.d1b])
        assert ave.data[1, 0] == 1
        assert ave['obs3'][0] == 5.
        npt.assert_almost_equal(ave['obs1'][2], 4.5)
        # CQ-7: the averaged Data must carry over headers and indvar (not just
        # cols), so downstream consumers (constraints read .indvar, save reads
        # .headers) keep working on averaged/smoothed results.
        assert ave.headers == self.d1.headers
        assert ave.indvar == self.d1.indvar == 'x'

    def test_from_columns(self):
        # CQ-3: shared factory used by the simulator backends to assemble
        # scan/time-course Data. Sets cols, headers, indvar (defaults to col 0).
        arr = np.array([[0., 3., 5.], [1., 2., 6.]])
        d = data.Data.from_columns(arr, ['time', 'obs1', 'obs3'])
        assert d.cols == {'time': 0, 'obs1': 1, 'obs3': 2}
        assert d.headers == {0: 'time', 1: 'obs1', 2: 'obs3'}
        assert d.indvar == 'time'
        npt.assert_array_equal(d['obs3'], np.array([5., 6.]))

    def test_from_columns_explicit_indvar(self):
        arr = np.array([[1., 9.]])
        d = data.Data.from_columns(arr, ['kf', 'A'], indvar='kf')
        assert d.indvar == 'kf'

    def test_rename_column(self):
        # ADR-0028 Chunk 4: rename a data column header in place, rewiring both
        # cols (header->idx) and headers (idx->header); the array is untouched, so
        # the renamed column reads the same numbers.
        d = data.Data.from_columns(np.array([[0., 3., 5.], [1., 2., 6.]]),
                                   ['time', 'obs1', 'obs3'])
        d.rename_column('obs1', 'pErk')
        assert d.cols == {'time': 0, 'pErk': 1, 'obs3': 2}
        assert d.headers == {0: 'time', 1: 'pErk', 2: 'obs3'}
        npt.assert_array_equal(d['pErk'], np.array([3., 2.]))

    def test_rename_column_to_same_name_is_noop(self):
        d = data.Data.from_columns(np.array([[0., 3.]]), ['time', 'obs1'])
        d.rename_column('obs1', 'obs1')
        assert d.cols == {'time': 0, 'obs1': 1}

    @raises(printing.PybnfError)
    def test_rename_missing_column_raises(self):
        d = data.Data.from_columns(np.array([[0., 3.]]), ['time', 'obs1'])
        d.rename_column('nope', 'pErk')

    @raises(printing.PybnfError)
    def test_rename_to_existing_column_raises(self):
        # Renaming onto an existing different column would silently merge two columns.
        d = data.Data.from_columns(np.array([[0., 3., 5.]]), ['time', 'obs1', 'obs3'])
        d.rename_column('obs1', 'obs3')

    @raises(printing.PybnfError)
    def test_rename_indvar_raises(self):
        # Remapping the independent variable (column 0) would corrupt the time/scan axis.
        d = data.Data.from_columns(np.array([[0., 3.]]), ['time', 'obs1'])
        d.rename_column('time', 't')

    def test_whitespace(self):
        d = data.Data()
        d.data = d._read_file_lines(self.data2, r'\s+')
        assert d['obs1'][1] == 2

    @raises(printing.PybnfError)
    def test_misformatted(self):
        d = data.Data()
        d._read_file_lines(self.data3, r'\s+')

    def test_normalize(self):
        d0 = data.Data()
        d0.data = d0._read_file_lines(self.data0, r'\s+')
        d0.normalize('peak')
        npt.assert_allclose(d0.data, np.array([[0., 1., 1., 1., np.nan, 1.], [1., 1., 1., 1., np.nan, 1.]]))

    def test_normalize_column_specific_leaves_other_columns(self):
        """Normalization restricted to specific columns should not touch other columns (issue #276)."""
        d = data.Data()
        d.data = d._read_file_lines(self.data1, r'\s+')
        original_obs3 = d.data[:, 3].copy()
        # Normalize only obs1 (column name), leaving obs2 and obs3 untouched
        d.normalize([('peak', ['obs1'])])
        # obs1 should be normalized (max is 4, so values become 3/4, 2/4, 4/4)
        npt.assert_allclose(d.data[:, 1], np.array([3./4, 2./4, 4./4]))
        # obs3 should be untouched
        npt.assert_allclose(d.data[:, 3], original_obs3)

    def test_result_normalize_column_specific(self):
        """Result.normalize with column-specific settings should only normalize listed columns (issue #276).

        This tests the fix for the edge case where a .prop file shares a suffix with a .exp file:
        normalization should only affect columns present in the .exp file.
        """
        # Simulate sim data with columns: time, obs1, obs2, obs3
        # where obs1 appears in the .exp file but obs3 only appears in a .prop constraint
        sim = data.Data()
        sim.data = sim._read_file_lines(self.data1, r'\s+')
        original_obs3 = sim.data[:, 3].copy()

        simdata = {'model1': {'data_suffix': sim}}
        res = algorithms.Result(None, simdata, 'test')

        # Column-specific normalization: only normalize obs1 (as the fix would produce)
        settings = {'data_suffix': [('peak', ['obs1'])]}
        res.normalize(settings)

        # obs1 should be normalized
        npt.assert_allclose(sim.data[:, 1], np.array([3./4, 2./4, 4./4]))
        # obs3 should be untouched — this is the key assertion for issue #276
        npt.assert_allclose(sim.data[:, 3], original_obs3)
