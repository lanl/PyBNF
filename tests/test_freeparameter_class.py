from .context import pset
from .context import printing
from nose.tools import raises

import numpy as np


class TestFreeParameter:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.p0 = pset.FreeParameter('var0__FREE__', 'normal_var', 0, 1)
        cls.p1 = pset.FreeParameter('var1__FREE__', 'lognormal_var', 1, 2)
        cls.p2 = pset.FreeParameter('var2__FREE__', 'loguniform_var', 0.01, 100)
        cls.p3 = pset.FreeParameter('var2__FREE__', 'uniform_var', 0, 10)
        cls.p4 = pset.FreeParameter('var2__FREE__', 'uniform_var', 0, 10, False)

    @classmethod
    def teardown_class(cls):
        pass

    def test_check_init(self):
        assert self.p0.value is None
        assert self.p0.type == 'normal_var'
        assert not self.p0.bounded

        assert not self.p1.bounded
        assert self.p1.lower_bound == 0.0
        assert np.isinf(self.p1.upper_bound)

        assert self.p2.upper_bound == 100

        assert self.p3.bounded
        assert not self.p4.bounded

    def test_check_assignment(self):
        self.p1.set_value(2)
        assert self.p1.value == 2.0

    @raises(printing.PybnfError)
    def test_check_erroneous_assignment(self):
        self.p3.set_value(11)

    def test_distribution(self):
        xs = [self.p3.sample_value(set_to_value=False).value for x in range(100000)]
        for x in xs:
            assert self.p3.lower_bound <= x < self.p3.upper_bound
        ys = [self.p0.sample_value(set_to_value=False).value for x in range(100000)]
        assert np.all(np.array(ys)>=0.0)

    def test_sample_value(self):
        self.p0.sample_value(set_to_value=True)
        assert self.p0.value is not None