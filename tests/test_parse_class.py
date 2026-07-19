import pytest

from pybnf.printing import PybnfError

from .context import parse


class TestParse:
    @classmethod
    def setup_class(cls):
        cls.s = ['output_dir =  world #test test', 'verbosity = 3', 'model = thing.bngl: data.exp', 'loguniform_var = derp 1 3',
                 ' #derp = derp', 'uniform_var = var__FREE 1 5', 'lognormal_var= var2__FREE 0.01 1.0e5',
                 'uniform_var = var3__FREE 4 5', 'model = another.bngl: d1.exp, d2.exp',
                 'credible_intervals=68 95 99.7', 'var=a 1 2', 'logvar=b 3',
                 'normalization=init : data1.exp, (data2.exp: 4,6-8), (data3.exp: var1,var2)',
                 'normalization=zero: (data2.exp:xyz)',
                 'cluster_type = slurm',
                 'initialization_distribution = bounds',
                 'random_seed = 12345',
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
        assert parse.parse(self.s[15]) == ['initialization_distribution', 'bounds']
        assert parse.parse(self.s[16]) == ['random_seed', '12345']
        assert parse.parse(self.s[17]) == ['time_course', 'model', 'thing.bngl', 'time', '100', 'step', '10']
        assert parse.parse(self.s[18]) == ['param_scan', 'model', 'another.bngl', 'param', 'var2__FREE', 'min', '10', 'max', '30', 'step', '1', 'time', '100']
        assert parse.parse(self.s[19]) == \
               ['mutant', 'another.bngl', 'm1', [['a4__FREE', '=', '42'], ['b5__FREE', '*', '17']], ['data1m1.exp', 'data2m1.exp']]

    def test_normalize_parse(self):
        assert parse.parse_normalization_def('init') == 'init'
        assert parse.parse_normalization_def('init: data1.exp') == {'data1.exp': 'init'}
        assert parse.parse_normalization_def('init: ( data1.exp: 1,5-8 )') == {'data1.exp': ('init', [1, 5, 6, 7, 8])}
        assert parse.parse_normalization_def('init : (data1.exp: VAR_1, XXX)') == {'data1.exp': ('init', ['VAR_1', 'XXX'])}
        assert parse.parse_normalization_def('init : ( data1.exp: VAR_1, XXX ) , data2.exp') == {'data1.exp': ('init', ['VAR_1', 'XXX']), 'data2.exp': 'init'}

    def test_normalization_chain_parse(self):
        # ADR-0066 (#479): a single argument-less token stays a bare string (backward-compatible);
        # a floor arg defaults to 0.03; a comma-separated chain is a list of transforms.
        assert parse.parse_normalization_chain('peak') == 'peak'
        assert parse.parse_normalization_chain('scale') == 'scale'
        assert parse.parse_normalization_chain('floor') == [('floor', 0.03)]
        assert parse.parse_normalization_chain('floor 0.05') == [('floor', 0.05)]
        assert parse.parse_normalization_chain('floor 0.03, scale') == [('floor', 0.03), 'scale']
        assert parse.parse_normalization_chain('floor 0.03, peak') == [('floor', 0.03), 'peak']
        # A non-numeric floor argument is a clear error, not a silent mis-parse.
        with pytest.raises(PybnfError):
            parse.parse_normalization_chain('floor abc')

    def test_normalization_whole_fit_chain(self):
        # The whole-fit (no ':') form routes through the chain parser too.
        assert parse.parse_normalization_def('floor 0.03, scale') == [('floor', 0.03), 'scale']
        assert parse.parse_normalization_def('peak') == 'peak'

    def test_normalization_obs_chain_ploop(self):
        # Per-observable + whole-fit chains land under the right config keys.
        d = parse.ploop(['normalization x = floor 0.03, scale\n'])
        assert d[('normalization', 'x')] == [('floor', 0.03), 'scale']
        d2 = parse.ploop(['normalization egf.y = floor\n'])
        assert d2[('normalization', 'egf.y')] == [('floor', 0.03)]
        d3 = parse.ploop(['normalization = floor 0.03, scale\n'])
        assert d3['normalization'] == [('floor', 0.03), 'scale']
        # The legacy single-token forms are byte-identical.
        assert parse.ploop(['normalization x = peak\n'])[('normalization', 'x')] == 'peak'
        assert parse.ploop(['normalization = peak\n'])['normalization'] == 'peak'

    def test_capital(self):
        assert parse.parse('Model = string.bngl: string.exp') == ['model', 'string.bngl', 'string.exp']
        assert parse.parse('Output_dir = string') == ['output_dir', 'string']
        assert parse.parse('vErbosity = 2') == ['verbosity', '2']

    def test_punctuation(self):
        assert parse.parse('bng_command = some/crazy!!-folder$$=\\"/BNG2.pl') == ['bng_command',
                                                                                  'some/crazy!!-folder$$=\\"/BNG2.pl']
        assert parse.parse('bngl_backend = bngsim') == ['bngl_backend', 'bngsim']

    def test_precondition_adapt_key(self):
        # p_dream's one extra knob (PDreamConfig.precondition_adapt) must parse as an
        # integer config key -- it was a schema field with no grammar entry, so a conf
        # line for it used to raise a ParseException (tutorial lesson 40).
        assert parse.parse('precondition_adapt = 250') == ['precondition_adapt', '250']

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
        assert d['initialization_distribution'] == 'bounds'
        assert d['random_seed'] == 12345
        assert d['time_course'] == [{'model': 'thing.bngl', 'time': '100', 'step': '10'}]
        assert d['param_scan'] == [{'model': 'another.bngl', 'param': 'var2__FREE', 'min': '10', 'max': '30', 'step': '1', 'time': '100'}]
        assert d['mutant'] == [['another.bngl', 'm1', [['a4__FREE', '=', '42'], ['b5__FREE', '*', '17']], ['data1m1.exp', 'data2m1.exp']]]
        assert 'data2m1.exp' in d['exp_data']

        d2 = parse.ploop(['credible_intervals=68'])
        assert d2['credible_intervals'] == [68.0]

        d3 = parse.ploop(['normalization=zero'])
        assert d3['normalization'] == 'zero'

    def test_postprocess_multiple_lines_consistent(self):
        # Each postprocess line should parse to a flat [script, *suffixes] list.
        # The 2nd+ line used to be double-wrapped (append([values]) instead of
        # append(values)), so _load_postprocessing saw a list-of-list as the
        # script path and empty suffixes for every postprocess line after the first.
        d = parse.ploop(['postprocess = s1.py sufA sufB', 'postprocess = s2.py sufC'])
        assert d['postprocess'] == [['s1.py', 'sufA', 'sufB'], ['s2.py', 'sufC']]
        # A single line is unchanged.
        d1 = parse.ploop(['postprocess = only.py sufA'])
        assert d1['postprocess'] == [['only.py', 'sufA']]

    def test_noise_model_grammar(self):
        # noise_model <obs> = <family>, <param> = <verb> <arg>[, ...][, location = mean|median] (ADR-0021, ADR-0024)
        assert parse.parse('noise_model obs2 = laplace, scale = fit b_obs2__FREE') == \
            ['noise_model', 'obs2', 'laplace', ['scale', 'fit', 'b_obs2__FREE']]
        assert parse.parse('noise_model obs3 = normal, sigma = read_exp_file _SD') == \
            ['noise_model', 'obs3', 'normal', ['sigma', 'read_exp_file', '_SD']]
        # forward-compatible: several "<param> = <verb> <arg>" fields on one line
        assert parse.parse('noise_model obs5 = student_t, scale = fit s__FREE, df = fix_at 4') == \
            ['noise_model', 'obs5', 'student_t', ['scale', 'fit', 's__FREE'], ['df', 'fix_at', '4']]
        # the optional location field (ADR-0024)
        assert parse.parse('noise_model obs4 = lognormal, sigma = read_exp_file _SD, location = mean') == \
            ['noise_model', 'obs4', 'lognormal', ['sigma', 'read_exp_file', '_SD'], ['location', 'mean']]

    def test_noise_model_ploop(self):
        d = parse.ploop(['objfunc = chi_sq',
                         'noise_model obs2 = laplace, scale = fit b_obs2__FREE',
                         'noise_model obs3 = normal, sigma = read_exp_file _SD, location = mean'])
        assert d[('noise_model', 'obs2')] == ('laplace', {'scale': ('fit', 'b_obs2__FREE')}, None)
        assert d[('noise_model', 'obs3')] == ('normal', {'sigma': ('read_exp_file', '_SD')}, 'mean')

    def test_catalog_var_keyword_arity(self):
        # The v2 catalog families (#417): the two-parameter families keep the two-number
        # form, the one-parameter unbounded families (exponential/chisquare/rayleigh) take a
        # single number, and the arity is enforced at parse time (a clean error, not a
        # downstream TypeError).
        assert parse.parse('cauchy_var = k 0 2') == ['cauchy_var', 'k', '0', '2']
        assert parse.parse('gamma_var = k 2 3') == ['gamma_var', 'k', '2', '3']
        assert parse.parse('exponential_var = k 0.5') == ['exponential_var', 'k', '0.5']
        assert parse.parse('chisquare_var = k 4') == ['chisquare_var', 'k', '4']
        assert parse.parse('rayleigh_var = k 1.5') == ['rayleigh_var', 'k', '1.5']
        # The arity is enforced at parse time -- ploop surfaces it as a clean PybnfError.
        with pytest.raises(PybnfError):
            parse.ploop(['normal_var = k 5'])         # a two-parameter family needs two numbers
        with pytest.raises(PybnfError):
            parse.ploop(['exponential_var = k 0.5 9'])  # a one-parameter family takes exactly one

    def test_node_parse(self):
        assert parse.parse('worker_nodes = cn196 192.168.1.1') == ['worker_nodes', 'cn196', '192.168.1.1']
        assert parse.parse('scheduler_node = this_machine') == ['scheduler_node', 'this_machine']

    def test_no_exp(self):
        assert parse.parse('model=thing.bngl: None') == ['model', 'thing.bngl']
        assert parse.parse('mutant = thing mutant a*2 b=0 : None') == ['mutant', 'thing', 'mutant', [['a', '*', '2'], ['b', '=', '0']], []]

    def test_model_declaration_grammar(self):
        # New-era `model:` declaration (ADR-0028): tagged 'model_decl' so it is
        # distinguishable from the legacy `model = file : exp` form, whose tokens are
        # otherwise shaped identically.
        assert parse.parse('model: egfr.bngl') == ['model_decl', 'egfr.bngl']
        assert parse.parse('model: egfr.bngl, mek1.xml, g.target') == \
            ['model_decl', 'egfr.bngl', 'mek1.xml', 'g.target']
        # The legacy `model = ...` form backtracks cleanly past the declaration grammar.
        assert parse.parse('model = egfr.bngl : d.exp') == ['model', 'egfr.bngl', 'd.exp']

    def test_model_declaration_ploop_accumulates(self):
        # Multiple `model:` lines union; each file folds like `model = file : none`
        # (added to the models set with an empty exp list); the declared files are
        # recorded in the structural 'model' marker for the config-layer edition gate.
        d = parse.ploop(['model: a.bngl', 'model: b.bngl, c.target'])
        assert d['models'] == {'a.bngl', 'b.bngl', 'c.target'}
        assert d['a.bngl'] == [] and d['b.bngl'] == [] and d['c.target'] == []
        assert d['exp_data'] == set()          # declarations bind no data
        assert d['model'] == ['a.bngl', 'b.bngl', 'c.target']

    def test_condition_grammar(self):
        # New-era `condition:` (ADR-0028): name + perturbations (var op val), with an
        # optional `model:` ref. The perturbations are the last group; the model ref
        # (when present) sits at l[2], making the line len 4 vs 3.
        assert parse.parse('condition: dimer_dead, perturbations: kdimer = 0') == \
            ['condition', 'dimer_dead', [['kdimer', '=', '0']]]
        assert parse.parse('condition: oe, model: erbb2.bngl, perturbations: a * 20, b / 2') == \
            ['condition', 'oe', ['erbb2.bngl'], [['a', '*', '20'], ['b', '/', '2']]]

    def test_condition_species_perturbation_grammar(self):
        # A SPECIES perturbation (#474): a QUOTED BNGL pattern (carries commas) = value, where the
        # value is a number OR a param-expression -- emitted as setConcentration (a wash/bolus).
        # A quoted LHS routes to the species op; a bare-id LHS stays the parameter op. The pattern
        # and the expression are captured verbatim (commas inside are protected by the quotes).
        assert parse.parse('condition: wash, perturbations: "IGF1(ds,hs,label~hot)" = 0') == \
            ['condition', 'wash', [['IGF1(ds,hs,label~hot)', '=', '0']]]
        assert parse.parse(
            'condition: w, perturbations: hot_conc = 7e-12, "IGF1(ds,hs,label~cold)" = c*(NA*V)') == \
            ['condition', 'w', [['hot_conc', '=', '7E-12'], ['IGF1(ds,hs,label~cold)', '=', 'c*(NA*V)']]]

    def test_condition_parameter_reference_value_grammar(self):
        # A per-condition estimated initial condition (ADR-0076): the perturbation value is a
        # bare identifier naming a free parameter (`I0_ = I0_CA`), not a number. The identifier
        # alternative parses as the third token of the (var, op, val) group, and a number and a
        # parameter reference coexist in one condition (config.py routes each by value type).
        assert parse.parse('condition: uCA, perturbations: I0_ = I0_CA') == \
            ['condition', 'uCA', [['I0_', '=', 'I0_CA']]]
        assert parse.parse('condition: c, perturbations: N_ = 39560000, I0_ = I0_CA') == \
            ['condition', 'c', [['N_', '=', '39560000'], ['I0_', '=', 'I0_CA']]]

    def test_condition_ploop_tuple_key(self):
        d = parse.ploop(['condition: c1, perturbations: kf = 1e-3, kr - 2',
                         'condition: c2, model: m.bngl, perturbations: a / 10'])
        # The num grammar normalizes the exponent marker to uppercase E.
        assert d[('condition', 'c1')] == (None, [('kf', '=', '1E-3'), ('kr', '-', '2')])
        assert d[('condition', 'c2')] == ('m.bngl', [('a', '/', '10')])

    def test_condition_duplicate_name_raises(self):
        with pytest.raises(PybnfError, match="Condition 'c' is specified multiple times"):
            parse.ploop(['condition: c, perturbations: a = 1', 'condition: c, perturbations: b = 2'])

    def test_experiment_grammar(self):
        # New-era `experiment:` (ADR-0028): a name + a required `data:` list, plus the
        # optional condition/model/type/method labeled sub-fields, each a pp.Group.
        assert parse.parse('experiment: egf_high, data: hi_r1.exp, hi_r2.exp') == \
            ['experiment', 'egf_high', ['data', 'hi_r1.exp', 'hi_r2.exp']]
        assert parse.parse('experiment: egf_dd, condition: dimer_dead, data: dd.exp') == \
            ['experiment', 'egf_dd', ['condition', 'dimer_dead'], ['data', 'dd.exp']]

    def test_experiment_fields_order_independent(self):
        # pp.Each lets the labeled fields appear in any order after the name; only data
        # is required. ploop reads them by label, so order never matters.
        d = parse.ploop(['experiment: full, type: time_course, data: a.exp, b.exp, '
                         'condition: c1, model: m.xml, method: ssa'])
        assert d[('experiment', 'full')] == {
            'type': 'time_course', 'data': ['a.exp', 'b.exp'],
            'condition': 'c1', 'model': 'm.xml', 'method': 'ssa'}

    def test_experiment_t_end_field(self):
        # ADR-0046: a parameter_scan's optional `t_end:` fixed endpoint -- a number read by
        # label (order-independent), absent => the scan runs to steady state.
        d = parse.ploop(['experiment: dose, type: parameter_scan, t_end: 500, data: d.exp'])
        assert d[('experiment', 'dose')] == {
            'type': 'parameter_scan', 't_end': '500', 'data': ['d.exp']}

    def test_experiment_ploop_stages_data_files(self):
        # Replicate data files are also staged into the exp_data set so the
        # normalization key can validate against them (as model/mutant lines do).
        d = parse.ploop(['experiment: e, data: r1.exp, r2.exp'])
        assert d[('experiment', 'e')] == {'data': ['r1.exp', 'r2.exp']}
        assert {'r1.exp', 'r2.exp'} <= d['exp_data']

    def test_experiment_requires_data(self):
        # data: is the one required field -- a bare experiment errors (with the
        # experiment-specific format hint).
        with pytest.raises(PybnfError, match='experiment:'):
            parse.ploop(['experiment: e, condition: c'])

    def test_experiment_duplicate_name_raises(self):
        with pytest.raises(PybnfError, match="Experiment 'x' is specified multiple times"):
            parse.ploop(['experiment: x, data: a.exp', 'experiment: x, data: b.exp'])

    def test_observable_grammar(self):
        # New-era `observable:` (ADR-0028, Chunk 4): a column-header override mapping a
        # model entity (the key) to a differently-named data column header (the value).
        assert parse.parse('observable: pErk, column: pErk_measured') == \
            ['observable', 'pErk', 'pErk_measured']

    def test_observable_ploop_tuple_key(self):
        # The override becomes a structural ('observable', entity) tuple key -> header
        # (like a condition / noise_model key), so config.py edition-gates it and the
        # golden config tests pass it through silently (a non-string key is never unused).
        d = parse.ploop(['observable: pErk, column: pErk_measured',
                         'observable: pAkt, column: pAkt_obs'])
        assert d[('observable', 'pErk')] == 'pErk_measured'
        assert d[('observable', 'pAkt')] == 'pAkt_obs'

    def test_observable_duplicate_entity_raises(self):
        with pytest.raises(PybnfError, match="Observable 'pErk' is specified multiple times"):
            parse.ploop(['observable: pErk, column: a', 'observable: pErk, column: b'])

    def test_observable_missing_column_errors(self):
        # column: is the one required field -- a bare observable line errors with the
        # observable-specific format hint.
        with pytest.raises(PybnfError, match='observable:'):
            parse.ploop(['observable: pErk'])


class TestParameterRecord:
    """The new-era ``parameter:`` record grammar (ADR-0043): a fully-labeled
    free-parameter declaration parsed into ordered named string fields. config.py
    validates the field set against the family and builds the FreeParameter."""

    def test_parses_into_ordered_named_fields(self):
        assert parse.parse('parameter: k, prior: normal, mean: 0, sd: 1, lower: -5, upper: 5') == \
            ['parameter', 'k', ['prior', 'normal'], ['mean', '0'], ['sd', '1'],
             ['lower', '-5'], ['upper', '5']]
        # a bare-word value (a family name / a space) and a numeric value coexist
        assert parse.parse('parameter: k, prior: normal, space: log10, mean: 1, sd: 0.5') == \
            ['parameter', 'k', ['prior', 'normal'], ['space', 'log10'], ['mean', '1'], ['sd', '0.5']]
        # the head alone with a single field
        assert parse.parse('parameter: k, initial_value: 5') == \
            ['parameter', 'k', ['initial_value', '5']]

    def test_ploop_stores_fields_dict(self):
        d = parse.ploop(['parameter: k, prior: normal, mean: 0, sd: 1, lower: -5, upper: 5',
                         'parameter: j, prior: uniform, lower: 0, upper: 10'])
        assert d[('parameter', 'k')] == {'prior': 'normal', 'mean': '0', 'sd': '1',
                                         'lower': '-5', 'upper': '5'}
        assert d[('parameter', 'j')] == {'prior': 'uniform', 'lower': '0', 'upper': '10'}

    def test_duplicate_field_raises(self):
        with pytest.raises(PybnfError, match="field 'mean' is specified multiple times"):
            parse.ploop(['parameter: k, prior: normal, mean: 0, mean: 1'])

    def test_duplicate_parameter_raises(self):
        with pytest.raises(PybnfError, match="Parameter 'k' is specified multiple times"):
            parse.ploop(['parameter: k, prior: normal, mean: 0, sd: 1',
                         'parameter: k, prior: uniform, lower: 0, upper: 1'])

    def test_malformed_line_reports_parameter_hint(self):
        # A field missing its value -> the parameter-specific format hint.
        with pytest.raises(PybnfError, match='parameter:'):
            parse.ploop(['parameter: k, prior'])
