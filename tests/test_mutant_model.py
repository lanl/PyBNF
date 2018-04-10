from .context import pset
from os import remove
import numpy as np


class TestMutantModel:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.file1 = 'bngl_files/Simple.bngl'

        cls.dict1 = {'kase__FREE__': 3.8, 'pase__FREE__': 0.16, 'koff__FREE__': 4.4e-3}
        cls.mutations = ['kase__FREE__+=pase__FREE__', 'koff__FREE__=0']

    @classmethod
    def teardown_class(cls):
        pass

    def test_mutate(self):
        model1 = pset.BNGLModel(self.file1)
        assert model1.param_names == ('kase__FREE__', 'koff__FREE__', 'pase__FREE__')
        model_mut = pset.MutantModel('testmodel', model1, self.mutations)
        ps = pset.PSet(self.dict1)
        model_mut_copy = model_mut.copy_with_param_set(ps)
        assert model_mut_copy.model.param_set['koff__FREE__'] == 0
        np.testing.assert_almost_equal(model_mut_copy.model.param_set['kase__FREE__'], 3.96)
        np.testing.assert_almost_equal(model_mut_copy.model.param_set['pase__FREE__'], 0.16)