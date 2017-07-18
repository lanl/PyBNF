import math
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

    def test_number_reader(self):
        assert data.Data.to_number(self.str0) == math.inf
        assert not math.isfinite(data.Data.to_number(self.str1))
        assert math.isnan(data.Data.to_number(self.str2))
        assert data.Data.to_number(self.str3) == 6.022e-23

    @raises(ValueError)
    def test_number_reader_failure(self):
        data.Data.to_number(self.str4)