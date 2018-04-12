from .context import pset, printing
from nose.tools import raises
import os
import xml.etree.ElementTree as ET
import re
import numpy as np
import shutil


class TestSbmlModel:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.file = 'bngl_files/raf.xml'
        cls.savefile = 'raf_test'
        cls.savefile2 = 'raf_test_exec'
        cls.params = {'K3':8000., 'K5': 0.3}
        cls.folder = 'raf_test'
        cls.folder_scan = 'raf_test_scan'
        try:
            shutil.rmtree(cls.folder)
        except OSError:
            pass
        try:
            shutil.rmtree(cls.folder_scan)
        except OSError:
            pass

    @classmethod
    def teardown_class(cls):
        try:
            os.remove('%s.xml' % cls.savefile)
        except OSError:
            pass
        try:
            shutil.rmtree(cls.folder)
        except OSError:
            pass
        try:
            shutil.rmtree(cls.folder_scan)
        except OSError:
            pass

    def test_init(self):
        m = pset.SbmlModel(self.file)
        assert m.name == 'raf'
        assert m.file == self.file

    def test_execute(self):
        os.mkdir(self.folder)
        fullpath = os.getcwd()+'/'+self.folder
        ps = pset.PSet(self.params)
        action = pset.TimeCourse({'time': '1000', 'step': '10'})
        m = pset.SbmlModel(self.file, pset=ps, actions=(action,))
        result = m.execute(fullpath, self.savefile2, 1000)
        dat = result['time_course']
        assert abs(dat['RIRI'][-1] - 2.94514) < 0.01
        assert abs(dat['R'][-1] - 0.358949) < 0.01
        assert dat.cols['time'] == 0

    def test_param_scan(self):
        os.mkdir(self.folder_scan)
        fullpath = os.getcwd() + '/' + self.folder_scan
        ps = pset.PSet(self.params)
        action = pset.ParamScan({'param': 'K3', 'min': '500', 'max': '10000', 'step': '500', 'time': '1000'})
        m = pset.SbmlModel(self.file, pset=ps, actions=(action,))
        result = m.execute(fullpath, self.savefile2, 1000)
        dat = result['param_scan']
        assert dat.indvar == 'K3'
        assert abs(dat['I'][0] - 0.236666) < 0.01
        assert abs(dat['R'][-1] - 0.315964) < 0.01

    @raises(printing.PybnfError)
    def test_missing_key(self):
        pset.TimeCourse({'model': 'm'})

    @raises(printing.PybnfError)
    def test_extra_key(self):
        pset.ParamScan({'param': 'K3', 'min': '500', 'max': '10000', 'step': '500', 'time': '1000', 'xxx': '0'})

    @raises(printing.PybnfError)
    def test_invalid_num(self):
        pset.ParamScan({'param': 'K3', 'min': '500', 'max': '1000f', 'step': '500', 'time': '1000'})

