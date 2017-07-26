import numpy as np
from .context import pset
from nose.tools import raises


class TestPSet:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.dict2 = {'x': 1.0, 'y': 2.0, 'z': 3.14}
        cls.dict3 = {'xx': 3.0, 'yy': 700.3, 'ww': 5.2e-4, 'kk': 5.2e-39}
        cls.dict4 = {'kk': 5.2e-39, 'xx': 3.00, 'yy': 700.3, 'ww': 52e-5}
        cls.dict5 = {'x': 1.0, 'y': 2.0, 'z': 3.141}

        cls.bad_dict1 = {'x': 1.0, 'y': -2.0, 'z': 3.14}
        cls.bad_dict2 = {'x': 1.0, 'y': 2, 'z': 3.141}
        cls.bad_dict3 = {'x': 1.0, 42: 2.0, 'z': 3.141}
        cls.bad_dict4 = {'x': 1.0, 'y': np.nan, 'z': 3.14}
        cls.bad_dict5 = {'x': 1.0, 'y': np.inf, 'z': 3.14}


    def test_initialization(self):
        ps1 = pset.PSet(self.dict2)
        assert ps1['x'] == 1.0
        assert ps1['z'] == 3.14

    def test_keys_to_string(self):
        ps1 = pset.PSet(self.dict2)
        assert ps1.keys_to_string() == 'x\ty\tz'
        ps2 = pset.PSet(self.dict3)
        assert ps2.keys_to_string() == 'kk\tww\txx\tyy'

    def test_values_to_string(self):
        ps1 = pset.PSet(self.dict2)
        assert ps1.values_to_string() == '1.0\t2.0\t3.14'
        ps2 = pset.PSet(self.dict3)
        assert ps2.values_to_string() == '5.2e-39\t0.00052\t3.0\t700.3'

    def test_get_id(self):
        ps1a = pset.PSet(self.dict2)
        ps1b = pset.PSet(self.dict2)
        ps2a = pset.PSet(self.dict3)
        ps2b = pset.PSet(self.dict4)
        ps3 = pset.PSet(self.dict5)
        assert ps1a.get_id() == ps1b.get_id()
        assert ps2a.get_id() == ps2b.get_id()
        assert ps1a.get_id() != ps2a.get_id()
        assert ps3.get_id() != ps1a.get_id()

    @raises(TypeError)
    def test_immutable(self):
        ps1 = pset.PSet(self.dict2)
        ps1['x']=1.5

    @raises(ValueError)
    def test_keystr(self):
        ps1 = pset.PSet(self.bad_dict1)

    @raises(TypeError)
    def test_valtype(self):
        ps1 = pset.PSet(self.bad_dict2)

    @raises(TypeError)
    def test_keytype(self):
        ps1 = pset.PSet(self.bad_dict3)

    @raises(ValueError)
    def test_nan(self):
        ps1 = pset.PSet(self.bad_dict4)

    @raises(ValueError)
    def test_inf(self):
        ps1 = pset.PSet(self.bad_dict5)