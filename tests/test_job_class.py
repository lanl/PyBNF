from .context import algorithms
from .context import pset
from os import chdir
from os import environ
from os import remove
from os.path import isfile


class TestJob(object):
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.model = pset.Model('bngl_files/Tricky.bngl')
        d = {
            'koff__FREE__': 0.1,
            '__koff2__FREE__': 0.1,
            'kase__FREE__': 1,
            'pase__FREE__': 1
        }
        cls.pset = pset.PSet(d)
        cls.bngpath = environ['BNGPATH']
        cls.job = algorithms.Job([cls.model], cls.pset, 1, cls.bngpath)

    @classmethod
    def teardown_class(cls):
        remove('bngl_files/Tricky_1.bngl')
        # remove('bngl_files/Tricky_1_thing*')
        # remove('bngl_files/Tricky_1_p1_5*')
        # remove('bngl_files/Tricky_1.net')

    def test_job_write(self):
        chdir('bngl_files')
        self.job._write_models()
        assert isfile('Tricky_1.bngl')
        chdir('../')