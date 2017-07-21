import math
import numpy as np
from .context import data
from nose.tools import raises


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

        cls.data0 = [
            '#          time    fullyBoundAg        Bound2Ag        Bound1Ag          freeAg    fullyBoundAb\n',
            ' 0.00000000e+00  1.20000000e+01  8.00000000e+00  6.00000000e+00  0.00000000e+00  1.60000000e+01\n',
            ' 1.00000000e+00  1.20000000e+01  8.00000000e+00  6.00000000e+00  0.00000000e+00  1.60000000e+01\n'
        ]
        cls.d = data.Data()
        cls.d.data = cls.d._read_file_lines(cls.data0, '\s+')

    def test_number_reader(self):
        assert data.Data.to_number(self.str0) == math.inf
        assert not math.isfinite(data.Data.to_number(self.str1))
        assert math.isnan(data.Data.to_number(self.str2))
        assert data.Data.to_number(self.str3) == 6.022e-23

    @raises(ValueError)
    def test_number_reader_failure(self):
        data.Data.to_number(self.str4)

    def test_read_file_lines(self):
        d = data.Data()
        loc_data = d._read_file_lines(self.data0, '\s+')
        d.data = loc_data
        assert d.cols['time'] == 0
        assert len(d.cols.keys()) == 6
        assert d.data.shape == (2, 6)
        assert d.data[0, 2] == 8

    def test_column_access(self):
        assert np.array_equal(self.d['time'], np.array([0, 1]))

    @raises(KeyError)
    def test_column_access_failure(self):
        self.d['thing']