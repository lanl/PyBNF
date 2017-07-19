import math
from .context import data
from nose.tools import raises


class TestPset:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.str0 = "inf"
        #etc..

    def test_number_reader(self):
        pass
        # assert function_call() == expected answer


    #@raises(ValueError)
    #def test_number_reader_failure(self):
    #    data.Data.to_number(self.str4)