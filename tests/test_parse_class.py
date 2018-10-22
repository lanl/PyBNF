import os
from .context import parse


class TestParse:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.s = ['output_dir =  world #test test', 'verbosity = 3', 'model = thing.bngl: data.exp', 'loguniform_var = derp 1 3',
                 ' #derp = derp', 'uniform_var = var__FREE 1 5', 'lognormal_var= var2__FREE 0.01 1.0e5',
                 'uniform_var = var3__FREE 4 5', 'model = another.bngl: d1.exp, d2.exp',
                 'credible_intervals=68 95 99.7', 'var=a 1 2', 'logvar=b 3',
                 'normalization=init : data1.exp, (data2.exp: 4,6-8), (data3.exp: var1,var2)',
                 'normalization=zero: (data2.exp:xyz)',
                 'cluster_type = slurm',
                 'time_course = model: thing.bngl, time: 100, step: 10',
                 'param_scan=model:another.bngl, param:var2__FREE, min:10, max:30, step:1, time:100',
                 'mutant=another.bngl m1 a4__FREE=42 b5__FREE*17 : data1m1.exp, data2m1.exp']

    @classmethod
    def teardown_class(cls):
        pass

    def test_grammar(self):
        assert parse.parse(self.s[0]) == ['output_dir', 'world']
        assert parse.parse(self.s[1]) == ['verbosity', '3']
        assert parse.parse(self.s[2]) == ['model', 'thing.bngl', 'data.exp']
        assert parse.parse(self.s[3]) == ['loguniform_var', 'derp', '1', '3']
        assert parse.parse(self.s[5]) == ['uniform_var', 'var__FREE', '1', '5']
        assert parse.parse(self.s[6]) == ['lognormal_var', 'var2__FREE', '0.01', '1.0E5']
        assert parse.parse(self.s[7]) == ['uniform_var', 'var3__FREE', '4', '5']
        assert parse.parse(self.s[8]) == ['model', 'another.bngl', 'd1.exp', 'd2.exp']
        assert parse.parse(self.s[9]) == ['credible_intervals', '68', '95', '99.7']
        assert parse.parse(self.s[10]) == ['var', 'a', '1', '2']
        assert parse.parse(self.s[11]) == ['logvar', 'b', '3']
        assert parse.parse(self.s[12]) == ['normalization', 'init : data1.exp, (data2.exp: 4,6-8), (data3.exp: var1,var2)']
        assert parse.parse(self.s[15]) == ['time_course', 'model', 'thing.bngl', 'time', '100', 'step', '10']
        assert parse.parse(self.s[16]) == ['param_scan', 'model', 'another.bngl', 'param', 'var2__FREE', 'min', '10', 'max', '30', 'step', '1', 'time', '100']
        assert parse.parse(self.s[17]) == \
               ['mutant', 'another.bngl', 'm1', [['a4__FREE', '=', '42'], ['b5__FREE', '*', '17']], ['data1m1.exp', 'data2m1.exp']]

    def test_normalize_parse(self):
        assert parse.parse_normalization_def('init') == 'init'
        assert parse.parse_normalization_def('init: data1.exp') == {'data1.exp': 'init'}
        assert parse.parse_normalization_def('init: ( data1.exp: 1,5-8 )') == {'data1.exp': ('init', [1, 5, 6, 7, 8])}
        assert parse.parse_normalization_def('init : (data1.exp: VAR_1, XXX)') == {'data1.exp': ('init', ['VAR_1', 'XXX'])}
        assert parse.parse_normalization_def('init : ( data1.exp: VAR_1, XXX ) , data2.exp') == {'data1.exp': ('init', ['VAR_1', 'XXX']), 'data2.exp': 'init'}

    def test_capital(self):
        assert parse.parse('Model = string.bngl: string.exp') == ['model', 'string.bngl', 'string.exp']
        assert parse.parse('Output_dir = string') == ['output_dir', 'string']
        assert parse.parse('vErbosity = 2') == ['verbosity', '2']

    def test_punctuation(self):
        assert parse.parse('bng_command = some/crazy!!-folder$$=\\"/BNG2.pl') == ['bng_command',
                                                                                  'some/crazy!!-folder$$=\\"/BNG2.pl']

    def test_ploop(self):
        d = parse.ploop(self.s)
        assert 'output_dir' in d.keys()
        assert 'verbosity' in d.keys()
        assert 'model' not in d.keys()
        assert ('lognormal_var', 'var2__FREE') in d.keys()
        assert ('uniform_var', 'var__FREE') in d.keys()
        assert ('uniform_var', 'var3__FREE') in d.keys()

        assert d['output_dir'] == 'world'
        assert d['verbosity'] == 3
        assert type(d['verbosity']) == int
        assert d['thing.bngl'] == ['data.exp']
        assert d[('loguniform_var', 'derp')] == [1., 3., True]
        assert d[('uniform_var', 'var3__FREE')] == [4., 5., True]
        assert d['another.bngl'] == ['d1.exp', 'd2.exp']
        assert d['models'] == {'thing.bngl', 'another.bngl'}
        assert d['credible_intervals'] == [68., 95., 99.7]
        assert d[('var', 'a')] == [1., 2.]
        assert d[('logvar', 'b')] == [3.]
        assert d['normalization'] == {'data1.exp': 'init', 'data2.exp': [('init', [4,6,7,8]), ('zero', ['xyz'])], 'data3.exp': [('init', ['var1', 'var2'])]}
        assert d['cluster_type'] == 'slurm'
        assert d['time_course'] == [{'model': 'thing.bngl', 'time': '100', 'step': '10'}]
        assert d['param_scan'] == [{'model': 'another.bngl', 'param': 'var2__FREE', 'min': '10', 'max': '30', 'step': '1', 'time': '100'}]
        assert d['mutant'] == [['another.bngl', 'm1', [['a4__FREE', '=', '42'], ['b5__FREE', '*', '17']], ['data1m1.exp', 'data2m1.exp']]]
        assert 'data2m1.exp' in d['exp_data']

        d2 = parse.ploop(['credible_intervals=68'])
        assert d2['credible_intervals'] == [68.0]

        d3 = parse.ploop(['normalization=zero'])
        assert d3['normalization'] == 'zero'

    def test_node_parse(self):
        assert parse.parse('worker_nodes = cn196 192.168.1.1') == ['worker_nodes', 'cn196', '192.168.1.1']
        assert parse.parse('scheduler_node = this_machine') == ['scheduler_node', 'this_machine']

    def test_no_exp(self):
        assert parse.parse('model=thing.bngl: None') == ['model', 'thing.bngl']
        assert parse.parse('mutant = thing mutant a*2 b=0 : None') == ['mutant', 'thing', 'mutant', [['a', '*', '2'], ['b', '=', '0']], []]
