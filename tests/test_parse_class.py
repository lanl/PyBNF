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

    def test_grammar(self):
        s = ['job_name =  world', 'verbosity = 3', 'model = strings strings', 'mutate = derp 1 3' ' #derp = derp']
        assert parse.parse(s[0]) == ['job_name', 'world']
        assert parse.parse(s[1]) == ['verbosity', '3']
        assert parse.parse(s[2]) == ['model', 'strings', 'strings']
        assert parse.parse(s[3]) == ['mutate', 'derp', '1', '3']
    
    def test_capital(self):
        assert parse.parse('Model = string string') == ['model', 'string', 'string']
        assert parse.parse('Job_name = string') == ['job_name', 'string']
        assert parse.parse('vErbosity = 2') == ['verbosity', '2']