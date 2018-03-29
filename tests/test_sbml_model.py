from .context import pset
import os
import xml.etree.ElementTree as ET
import re
import numpy as np



class TestSbmlModel:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.file = 'bngl_files/raf.xml'
        cls.savefile = 'raf_test'
        cls.params = {'K3':8000., 'K5': 0.3}

    @classmethod
    def teardown_class(cls):
        try:
            os.remove('%s.xml' % cls.savefile)
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


