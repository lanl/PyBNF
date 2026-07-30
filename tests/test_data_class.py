import math
import numpy as np
import numpy.testing as npt
import pytest
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

    def test_floor_normalization(self):
        # x' = x + rho*max(x) per column (rho = 0.1); the independent variable is untouched.
        # d1 columns: obs1 max 4 (+0.4), obs2 max 4 (+0.4), obs3 max 10 (+1.0).
        d1 = copy.deepcopy(self.d1)
        d1.normalize_to_floor(0.1)
        npt.assert_allclose(d1.data, np.array(
            [[0., 3.4, 4.4, 6.], [1., 2.4, 3.4, 7.], [2., 4.4, 2.4, 11.]]))
        # A subset of columns; the others keep their raw values.
        d1b = copy.deepcopy(self.d1)
        d1b.normalize_to_floor(0.1, cols=[1])
        npt.assert_allclose(d1b.data, np.array(
            [[0., 3.4, 4., 5.], [1., 2.4, 3., 6.], [2., 4.4, 2., 10.]]))

    def test_floor_records_rho_and_argmax(self):
        d1 = copy.deepcopy(self.d1)
        d1.normalize_to_floor(0.03, cols=[1])
        rec = d1.normalization['obs1']
        assert rec.method == 'floor'
        assert rec.rho == 0.03
        assert rec.scale == 4.0            # the column max
        assert rec.ref_row == 2            # argmax (obs1 peaks at row 2)

    def test_normalize_dispatches_floor_tuple_and_chains(self):
        # Data.normalize accepts a (name, arg) transform, bare or in an ordered chain.
        d1 = copy.deepcopy(self.d1)
        d1.normalize(('floor', 0.1))
        npt.assert_allclose(d1.data[:, 1], np.array([3.4, 2.4, 4.4]))
        # Chain floor(0.1) then peak on obs1: floor -> [3.4,2.4,4.4] (max 4.4), then /4.4.
        d1c = copy.deepcopy(self.d1)
        d1c.normalize([(('floor', 0.1), [1]), ('peak', [1])])
        npt.assert_allclose(d1c.data[:, 1], np.array([3.4, 2.4, 4.4]) / 4.4)

    def test_floor_peak_unit_are_nan_aware_on_sparse_columns(self):
        # #479 follow-up: a sparse multi-observable target carries NaN in the rows where an
        # observable is unmeasured (each observable has its own measurement times). The floor
        # (ADR-0066) is applied to the experimental data too, so a plain np.max there returns NaN
        # and poisons the whole column -> every point NaN -> silently dropped in scoring ->
        # objective 0.0 for every pset. floor/peak/unit must skip the NaNs and use the measured
        # points' max only. Column obs1 = [3, NaN, 4]: nanmax = 4 at row 2.
        sparse = np.array([[0., 3., 4., 5.],
                           [1., np.nan, 3., 6.],
                           [2., 4., np.nan, 10.]])

        def mk():
            d = data.Data()
            d.data = sparse.copy()
            d.headers = {0: 'x', 1: 'obs1', 2: 'obs2', 3: 'obs3'}
            return d

        # floor 0.1 on the sparse obs1: measured points floored by 0.1*nanmax(=4); NaN row stays NaN.
        df = mk(); df.normalize_to_floor(0.1, cols=[1])
        npt.assert_allclose(df.data[[0, 2], 1], np.array([3.4, 4.4]))
        assert np.isnan(df.data[1, 1])
        assert df.normalization['obs1'].scale == 4.0 and df.normalization['obs1'].ref_row == 2

        # peak on the sparse obs1: /nanmax(=4); NaN row stays NaN.
        dp = mk(); dp.normalize_to_peak(cols=[1])
        npt.assert_allclose(dp.data[[0, 2], 1], np.array([3. / 4., 1.0]))
        assert np.isnan(dp.data[1, 1])

        # unit-scale on the sparse obs1: baseline row 0 (=3), then /nanmax-after-baseline(=1); NaN
        # row stays NaN. obs1-3 = [0, NaN, 1] -> /1 -> [0, NaN, 1].
        du = mk(); du.normalize_to_unit_scale(cols=[1])
        npt.assert_allclose(du.data[[0, 2], 1], np.array([0.0, 1.0]))
        assert np.isnan(du.data[1, 1])

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


class TestOutputSensitivities:
    """The gradient-path payload attached to a simulated Data (#385/#447)."""

    @staticmethod
    def _payload():
        # n_times=2, two selectors, two params; distinct values per cell so a
        # mis-sliced column is caught.
        d_param = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)
        return data.OutputSensitivities(
            selectors=['observable:A', 'observable:B'],
            param_names=['k1', 'k2'],
            ic_species=[],
            d_param=d_param,
            d_ic=None,
        )

    def test_default_data_has_no_sensitivities(self):
        # A plain Data carries the additive attribute, defaulting None (scalar path).
        assert data.Data().output_sensitivities is None

    def test_slice_for_selects_the_right_column(self):
        payload = self._payload()
        npt.assert_array_equal(payload.slice_for('observable:A'),
                               payload.d_param[:, 0, :])
        npt.assert_array_equal(payload.slice_for('observable:B'),
                               payload.d_param[:, 1, :])

    def test_slice_for_unknown_selector_raises_keyerror(self):
        with pytest.raises(KeyError):
            self._payload().slice_for('observable:missing')

    def test_slice_for_uncomputed_axis_raises_valueerror(self):
        # IC axis was never computed for this payload.
        with pytest.raises(ValueError):
            self._payload().slice_for('observable:A', axis='ic')

    def test_slice_for_rejects_bad_axis(self):
        with pytest.raises(ValueError):
            self._payload().slice_for('observable:A', axis='bogus')


class TestStackScanSensitivities:
    """Stacking per-dose-point tensors into one dose-axis scan payload (#476)."""

    @staticmethod
    def _point(a_k1, a_k2, ic_a=None):
        # One per-point payload with n_times=2 (only the LAST row is stacked): the first
        # row is a decoy that must NOT appear in the stacked result, the second is the
        # equilibrium row carrying the given (k1, k2) sensitivities for selector A.
        d_param = np.array([[[-9.0, -9.0]], [[a_k1, a_k2]]])   # (2 times, 1 selector, 2 params)
        d_ic = None
        if ic_a is not None:
            d_ic = np.array([[[-9.0]], [[ic_a]]])              # (2 times, 1 selector, 1 ic)
        return data.OutputSensitivities(
            selectors=['observable:A'], param_names=['k1', 'k2'],
            ic_species=['S()'] if ic_a is not None else [],
            d_param=d_param, d_ic=d_ic)

    def test_stacks_final_row_down_dose_axis(self):
        stacked = data.stack_scan_sensitivities(
            [self._point(1.0, 2.0), self._point(3.0, 4.0), self._point(5.0, 6.0)])
        assert stacked.selectors == ['observable:A']
        assert stacked.param_names == ['k1', 'k2']
        # (n_doses=3, n_selectors=1, n_params=2), each dose row = that point's LAST row.
        assert stacked.d_param.shape == (3, 1, 2)
        npt.assert_array_equal(stacked.d_param[:, 0, :],
                               [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert stacked.d_ic is None

    def test_stacks_ic_axis_when_present(self):
        stacked = data.stack_scan_sensitivities(
            [self._point(1.0, 2.0, ic_a=0.5), self._point(3.0, 4.0, ic_a=0.7)])
        assert stacked.ic_species == ['S()']
        assert stacked.d_ic.shape == (2, 1, 1)
        npt.assert_array_equal(stacked.d_ic[:, 0, 0], [0.5, 0.7])

    def test_empty_list_returns_none(self):
        assert data.stack_scan_sensitivities([]) is None

    def test_any_missing_point_returns_none(self):
        # A single scalar-path (None) point makes the whole scan scalar-path -> no tensor.
        assert data.stack_scan_sensitivities(
            [self._point(1.0, 2.0), None, self._point(5.0, 6.0)]) is None


class TestStackRaggedScanSensitivities:
    """Dose points whose sensitivity column sets disagree (#525).

    bngsim decides per ``Result`` which global functions it can differentiate, and PyBNF
    requests only those, so one dose point of a scan can legitimately carry a different
    selector list from its siblings'. Stacking must align by selector name and survive the
    mismatch, or -- where no alignment is possible -- say which dose point and which shapes
    are at fault instead of failing inside ``numpy.stack``.
    """

    @staticmethod
    def _point(values, selectors=('observable:A', 'expression:f'),
               param_names=('k1', 'k2'), n_times=2, ic=None):
        # ``values[sel][p]`` is the LAST row's d(sel)/d(param p); earlier rows are decoys.
        rows = [np.array([[-9.0] * len(param_names)] * len(selectors))
                for _ in range(max(n_times - 1, 0))]
        if n_times:
            rows.append(np.array([[float(values[s][p]) for p in range(len(param_names))]
                                  for s in selectors]))
        d_param = (np.stack(rows, axis=0) if rows
                   else np.zeros((0, len(selectors), len(param_names))))
        d_ic = None
        if ic is not None:
            d_ic = np.stack([np.array([[float(ic[s])] for s in selectors])] * n_times,
                            axis=0)
        return data.OutputSensitivities(
            selectors=list(selectors), param_names=list(param_names),
            ic_species=['S()'] if ic is not None else [],
            d_param=d_param, d_ic=d_ic)

    def test_column_absent_at_one_dose_is_dropped_not_fatal(self, caplog):
        # Point 1 lost 'expression:f' (the backend declined to differentiate it at that
        # dose only). The scan keeps the column every point has, with each dose's own
        # values -- the pre-#525 code raised ValueError from numpy.stack here.
        stacked = data.stack_scan_sensitivities([
            self._point({'observable:A': [1.0, 2.0], 'expression:f': [10.0, 20.0]}),
            self._point({'observable:A': [3.0, 4.0]}, selectors=('observable:A',)),
            self._point({'observable:A': [5.0, 6.0], 'expression:f': [50.0, 60.0]}),
        ])
        assert stacked.selectors == ['observable:A']
        assert stacked.d_param.shape == (3, 1, 2)
        npt.assert_array_equal(stacked.d_param[:, 0, :],
                               [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert 'expression:f absent at dose point(s) 1' in caplog.text

    def test_permuted_selector_order_aligns_by_name(self):
        # Same columns, different order at point 1: aligning by name (not position) keeps
        # each selector's own sensitivities -- position-stacking would swap A and f here.
        stacked = data.stack_scan_sensitivities([
            self._point({'observable:A': [1.0, 2.0], 'expression:f': [10.0, 20.0]}),
            self._point({'expression:f': [30.0, 40.0], 'observable:A': [3.0, 4.0]},
                        selectors=('expression:f', 'observable:A')),
        ])
        assert stacked.selectors == ['observable:A', 'expression:f']
        npt.assert_array_equal(stacked.d_param[:, 0, :], [[1.0, 2.0], [3.0, 4.0]])
        npt.assert_array_equal(stacked.d_param[:, 1, :], [[10.0, 20.0], [30.0, 40.0]])

    def test_no_shared_column_returns_none(self):
        # Nothing is uniform across the scan -> no tensor at all (the gradient path then
        # reports one missing-tensor error for the experiment).
        assert data.stack_scan_sensitivities([
            self._point({'observable:A': [1.0, 2.0]}, selectors=('observable:A',)),
            self._point({'expression:f': [3.0, 4.0]}, selectors=('expression:f',)),
        ]) is None

    def test_ic_axis_dropped_when_a_later_dose_lacks_it(self, caplog):
        # An axis present at dose 0 but missing later is a uniform gap, not a crash: the
        # parameter axis still stacks, the IC axis is dropped whole.
        stacked = data.stack_scan_sensitivities([
            self._point({'observable:A': [1.0, 2.0]}, selectors=('observable:A',),
                        ic={'observable:A': 0.5}),
            self._point({'observable:A': [3.0, 4.0]}, selectors=('observable:A',)),
        ])
        assert stacked.d_ic is None
        npt.assert_array_equal(stacked.d_param[:, 0, :], [[1.0, 2.0], [3.0, 4.0]])
        assert 'dose point 1 carries no d_ic tensor' in caplog.text

    def test_disagreeing_param_axis_labels_name_the_dose_point(self):
        with pytest.raises(printing.PybnfError) as exc:
            data.stack_scan_sensitivities([
                self._point({'observable:A': [1.0, 2.0]}, selectors=('observable:A',)),
                self._point({'observable:A': [3.0, 4.0]}, selectors=('observable:A',),
                            param_names=('k1', 'k3')),
            ])
        assert 'dose point 1' in str(exc.value)
        assert "'k3'" in str(exc.value)

    def test_tensor_shape_contradicting_its_own_labels_reports_shapes(self):
        point = self._point({'observable:A': [1.0, 2.0]}, selectors=('observable:A',))
        bad = data.OutputSensitivities(
            selectors=['observable:A', 'expression:f'],   # claims 2 columns, tensor has 1
            param_names=['k1', 'k2'], ic_species=[], d_param=point.d_param)
        with pytest.raises(printing.PybnfError) as exc:
            data.stack_scan_sensitivities([point, bad])
        assert 'dose point 1' in str(exc.value)
        assert '(2, 1, 2)' in str(exc.value)          # the actual tensor shape
        assert 'expected (n_times, 2, 2)' in str(exc.value)

    def test_no_integrated_rows_reports_the_dose_point(self):
        with pytest.raises(printing.PybnfError) as exc:
            data.stack_scan_sensitivities([
                self._point({'observable:A': [1.0, 2.0]}, selectors=('observable:A',)),
                self._point({'observable:A': [0.0, 0.0]}, selectors=('observable:A',),
                            n_times=0),
            ])
        assert 'dose point 1' in str(exc.value)
        assert 'no integrated rows' in str(exc.value)
