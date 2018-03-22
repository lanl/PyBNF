from .context import pset
from .context import printing
from nose.tools import raises


class TestPSet:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.p0 = pset.FreeParameter('var0__FREE__', 'normal_var', 0, 1)
        cls.p1 = pset.FreeParameter('var1__FREE__', 'lognormal_var', 1, 2)
        cls.p2 = pset.FreeParameter('var2__FREE__', 'loguniform_var', 0.01, 100)
        cls.p3 = pset.FreeParameter('var2__FREE__', 'uniform_var', 0, 10)
        cls.p4 = pset.FreeParameter('var2__FREE__', 'uniform_var', 0, 10, False)
        cls.p0.set_value(1.0)
        cls.p1.set_value(0.1)
        cls.p2.set_value(99.0)
        cls.p3.set_value(5.0)

        cls.fps0 = [cls.p0, cls.p1, cls.p2]
        cls.fps1 = [cls.p3, cls.p4, cls.p2]
        cls.fps2 = [cls.p3, cls.p0, cls.p1]

    def test_initialization(self):
        ps1 = pset.PSet(self.fps0)
        assert ps1['var0__FREE__'] == 1.0
        assert ps1['var1__FREE__'] == 0.1
        assert ps1['var2__FREE__'] == 99.0

    @raises(printing.PybnfError)
    def test_faulty_initialization(self):
        ps2 = pset.PSet(self.fps1)

    def test_keys_to_string(self):
        ps1 = pset.PSet(self.fps0)
        assert ps1.keys_to_string() == 'var0__FREE__\tvar1__FREE__\tvar2__FREE__'

    def test_values_to_string(self):
        ps1 = pset.PSet(self.fps0)
        assert ps1.values_to_string() == '1.0\t0.1\t99.0'

    def test_get_id(self):
        ps1a = pset.PSet(self.fps0)
        ps1b = pset.PSet(self.fps0)
        ps2 = pset.PSet(self.fps2)
        assert ps1a.get_id() == ps1b.get_id()
        assert ps1a.get_id() != ps2.get_id()
        assert ps2.get_id() != ps1b.get_id()

    @raises(TypeError)
    def test_immutable(self):
        ps1 = pset.PSet(self.fps0)
        ps1['var0__FREE__'] = 1.5
