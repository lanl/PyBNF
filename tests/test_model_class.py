from .context import pset
from nose.tools import raises


class TestModel:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.file1 = 'bngl_files/Simple.bngl'
        cls.file2 = 'bngl_files/ParamsEverywhere.bngl'
        cls.file3 = 'bngl_files/Tricky.bngl'

    def test_initialize(self):
        model1 = pset.Model(self.file1)
        assert model1.param_names == ('kase__FREE__', 'koff__FREE__', 'pase__FREE__')

        model2 = pset.Model(self.file2)
        assert model2.param_names == (
        'Ag_tot_1__FREE__', 'kase__FREE__', 'koff__FREE__', 'kon__FREE__', 'pase__FREE__', 't_end__FREE__')

        model3 = pset.Model(self.file3)
        assert model3.param_names == ('__koff2__FREE__', 'kase__FREE__', 'koff__FREE__')
