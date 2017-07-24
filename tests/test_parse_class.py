import os
from .context import parse


class TestParse:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_parse(self):
        file = open("testfile.txt","w")
        file.write('fit_type =  world\n \n #derp = derp')
        file.close()
        assert parse.ploop("testfile.txt") == {'fit_type': 'world'}
        os.remove("testfile.txt")
