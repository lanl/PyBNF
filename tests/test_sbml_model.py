from .context import pset
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
        assert m.xml.getroot().tag[-4:] == 'sbml'

    def test_set_pset(self):
        ps = pset.PSet(self.params)
        m = pset.SbmlModel(self.file)
        m._set_param_set(ps)
        m.save(self.savefile)

        # Read the saved xml file, and check if the param values are right.
        xml = ET.parse(self.savefile + '.xml')
        root = xml.getroot()
        ns = re.search('(?<={).*(?=})', root.tag).group(0)  # Extract the namespace from the root
        space = {'sbml': ns}
        params = root.findall('sbml:model/sbml:listOfParameters/sbml:parameter', namespaces=space)
        num_found = 0
        for p in params:
            pname = p.get('name')
            if pname == 'K3':
                np.testing.assert_almost_equal(float(p.get('value')), 8000.)
                num_found += 1
            if pname == 'K5':
                np.testing.assert_almost_equal(float(p.get('value')), 0.3)
                num_found += 1
        assert num_found == 2
        assert set(m.species) == {'R', 'I', 'RR', 'RI', 'RIR', 'RIRI'}

    def test_execute(self):
        os.mkdir(self.folder)
        fullpath = os.getcwd()+'/'+self.folder
        ps = pset.PSet(self.params)
        action = pset.TimeCourse(1000, 10)
        m = pset.SbmlModel(self.file, pset=ps, actions=(action,))
        m.copasi_command = os.environ['COPASIDIR'] + '/bin/CopasiSE'
        result = m.execute(fullpath, self.savefile2, 1000)
        dat = result['time_course']
        assert abs(dat['RIRI'][-1] - 2.94514) < 0.01
        assert abs(dat['R'][-1] - 0.358949) < 0.01
        assert dat.cols['Time'] == 0

    def test_param_scan(self):
        os.mkdir(self.folder_scan)
        fullpath = os.getcwd() + '/' + self.folder_scan
        ps = pset.PSet(self.params)
        action = pset.ParamScan('K3', 500, 10000, 500, 1000, logspace=False)
        m = pset.SbmlModel(self.file, pset=ps, actions=(action,))
        m.copasi_command = os.environ['COPASIDIR'] + '/bin/CopasiSE'
        result = m.execute(fullpath, self.savefile2, 1000)
        dat = result['param_scan']
        assert dat.indvar == 'K3'
        assert abs(dat['I'][0] - 0.236666) < 0.01
        assert abs(dat['R'][-1] - 0.315964) < 0.01

