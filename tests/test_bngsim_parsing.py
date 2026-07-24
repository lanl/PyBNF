"""Unit tests for pybnf.bngsim_model.parsing (pure BNGL action-line parsing).

These exercise the text->dict/tuple parsing helpers directly, with NO bngsim
wheel required, so they run on the bngsim-less CI tier where the e2e bridge
suites are skipped. Before the bngsim_model package split these helpers had no
direct coverage (#408); the regex-heavy comma-splitting and the net-vs-NF
setConcentration disambiguation are the highest-value targets.
"""

import pytest

from pybnf.bngsim_model import parsing


# ---------------------------------------------------------------- continuations
@pytest.mark.parametrize(
    'raw, expected',
    [
        ('simulate({t_end=>10})', 'simulate({t_end=>10})'),
        ('simulate({t_end=>10, \\\n        n_steps=>5})', 'simulate({t_end=>10, n_steps=>5})'),
        ('a\\\nb\\\nc', 'abc'),
    ],
)
def test_collapse_action_line_continuations(raw, expected):
    assert parsing._collapse_action_line_continuations(raw) == expected


# -------------------------------------------------------------------- body/split
@pytest.mark.parametrize(
    'line, name, expected',
    [
        ('simulate({method=>"ode"})', 'simulate', 'method=>"ode"'),
        ('  parameter_scan({parameter=>"k"})  ', 'parameter_scan', 'parameter=>"k"'),
        ('simulate({a=>1})', 'parameter_scan', None),   # name mismatch -> None
        ('not_an_action', 'simulate', None),
    ],
)
def test_extract_action_body(line, name, expected):
    assert parsing._extract_action_body(line, name) == expected


@pytest.mark.parametrize(
    'text, expected',
    [
        ('a=>1,b=>2,c=>3', ['a=>1', 'b=>2', 'c=>3']),
        ('a=>1,b=>[1,2,3],c=>4', ['a=>1', 'b=>[1,2,3]', 'c=>4']),          # nested list comma ignored
        ('a=>"x,y",b=>2', ['a=>"x,y"', 'b=>2']),                           # quoted comma ignored
        ('f=>g(1,2),h=>3', ['f=>g(1,2)', 'h=>3']),                         # nested paren comma ignored
        ('  ,a=>1, ,b=>2,  ', ['a=>1', 'b=>2']),                           # blank items dropped
        ('', []),
    ],
)
def test_split_top_level_commas(text, expected):
    assert parsing._split_top_level_commas(text) == expected


@pytest.mark.parametrize(
    'value_text, expected',
    [
        ('42', '42'),               # scalars stay strings
        ('"ode"', 'ode'),           # quotes stripped
        ("'ssa'", 'ssa'),
        ('[1,2,3]', ['1', '2', '3']),
        ('["a", "b"]', ['a', 'b']),
        ('[]', []),
        ('[[1,2],[3,4]]', [['1', '2'], ['3', '4']]),   # nested lists recurse
        # Exponential literals survive as strings for float() downstream — BNGL
        # writes sample_times this way (adopted from bngsim, lanl/bngsim#45).
        ('[5e-1,1,1E1]', ['5e-1', '1', '1E1']),
    ],
)
def test_parse_action_value(value_text, expected):
    assert parsing._parse_action_value(value_text) == expected


# --------------------------------------------------------------- action dicts
def test_parse_simulate_action_scalars_and_lists():
    line = 'simulate({method=>"ode", t_end=>10, n_steps=>5, print_functions=>["a","b"]})'
    assert parsing._parse_simulate_action(line) == {
        'method': 'ode',
        't_end': '10',
        'n_steps': '5',
        'print_functions': ['a', 'b'],
    }


def test_parse_simulate_action_sample_times_with_other_params():
    # sample_times is the list key that actually matters to the bngsim bridge:
    # it must survive alongside scalars, and the parser must decline non-simulate
    # lines rather than returning a partial dict (adopted from bngsim,
    # lanl/bngsim#45).
    line = 'simulate({method=>"ode", sample_times=>[0,5,10], suffix=>"tc"})'
    assert parsing._parse_simulate_action(line) == {
        'method': 'ode',
        'sample_times': ['0', '5', '10'],
        'suffix': 'tc',
    }


@pytest.mark.parametrize(
    'parser, line',
    [
        (parsing._parse_simulate_action, "setParameter('k1', 0.5)"),
        (parsing._parse_parameter_scan_action, 'simulate({method=>ode})'),
        (parsing._parse_parameter_scan_action, '# comment'),
    ],
)
def test_action_parsers_decline_foreign_lines(parser, line):
    assert parser(line) is None


def test_parse_parameter_scan_action_canonical_keys():
    # The alias test above feeds param/min/max/time/logspace; this feeds the
    # canonical spellings plus the suffix/steady_state passthroughs the bngsim
    # scan path reads (adopted from bngsim, lanl/bngsim#45).
    line = (
        'parameter_scan({parameter=>"kf",method=>"ode",'
        'par_min=>0.001,par_max=>1.0,n_scan_pts=>5,'
        'suffix=>"dose",t_end=>1000,steady_state=>1})'
    )
    assert parsing._parse_parameter_scan_action(line) == {
        'parameter': 'kf',
        'method': 'ode',
        'par_min': '0.001',
        'par_max': '1.0',
        'n_scan_pts': '5',
        'suffix': 'dose',
        't_end': '1000',
        'steady_state': '1',
    }


def test_parse_action_dict_skips_malformed_items():
    # An item without '=>' is logged and skipped, not crashed on.
    assert parsing._parse_action_dict('simulate({method=>"ode", junk})', 'simulate') == {
        'method': 'ode',
    }


@pytest.mark.parametrize('parser', [parsing._parse_parameter_scan_action, parsing._parse_bifurcate_action])
def test_parameter_scan_key_aliases(parser):
    name = 'parameter_scan' if parser is parsing._parse_parameter_scan_action else 'bifurcate'
    line = f'{name}({{param=>"k", min=>0.1, max=>10, time=>100, logspace=>1, n_steps=>20}})'
    assert parser(line) == {
        'parameter': 'k',     # param   -> parameter
        'par_min': '0.1',     # min     -> par_min
        'par_max': '10',      # max     -> par_max
        't_end': '100',       # time    -> t_end
        'log_scale': '1',     # logspace-> log_scale
        'n_steps': '20',      # passthrough
    }


# ------------------------------------------------------- setParameter / numeric
@pytest.mark.parametrize(
    'line, expected',
    [
        ('setParameter("k", 5)', ('k', 5.0)),
        ('setParameter("k", 2*3)', ('k', 6.0)),
        ('setParameter("rate", sqrt(16))', ('rate', 4.0)),
        ('setParameter("k", not_a_number)', None),   # eval failure -> None
        ('notSetParameter("k", 5)', None),
    ],
)
def test_parse_set_parameter(line, expected):
    assert parsing._parse_set_parameter(line) == expected


# -------------------------------------- setConcentration: net vs NF vs deferred
@pytest.mark.parametrize(
    'line, expected',
    [
        ('setConcentration("A", 100)', ('A', 100.0)),
        ('setConcentration("A", 2*50)', ('A', 100.0)),
        # NF-style symbolic value: float()/eval fail -> None (handed to NF parser)
        ('setConcentration("A", "EGF_copy_number")', None),
    ],
)
def test_parse_set_concentration_net(line, expected):
    assert parsing._parse_set_concentration(line) == expected


@pytest.mark.parametrize(
    'line, expected',
    [
        # deferred form keeps the value as text for later model-namespace eval (#46)
        ('setConcentration("A", k*2)', ('A', 'k*2')),
        ('setConcentration("A", "k*2")', ('A', 'k*2')),    # quotes stripped
        ('setConcentration("A", 100)', ('A', '100')),
        ('not_set("A", 1)', None),
    ],
)
def test_parse_set_concentration_expr(line, expected):
    assert parsing._parse_set_concentration_expr(line) == expected


@pytest.mark.parametrize(
    'line, expected',
    [
        ('setConcentration("EGFR(l)", "EGF_copy_number")', ('EGFR(l)', 'EGF_copy_number')),
        ('setConcentration("X", 100)', ('X', '100')),
        ('addConcentration("X", 1)', None),
    ],
)
def test_parse_set_concentration_nf(line, expected):
    assert parsing._parse_set_concentration_nf(line) == expected


@pytest.mark.parametrize(
    'line, expected',
    [
        ('addConcentration("A", 5)', ('A', 5.0)),
        ('addConcentration("A", 1+1)', ('A', 2.0)),
        ('setConcentration("A", 5)', None),
    ],
)
def test_parse_add_concentration(line, expected):
    assert parsing._parse_add_concentration(line) == expected


# ------------------------------------------------------------------- predicates
@pytest.mark.parametrize(
    'predicate, matching, non_matching',
    [
        (parsing._is_reset_concentrations, 'resetConcentrations()', 'resetParameters()'),
        (parsing._is_reset_parameters, '  resetParameters()', 'resetConcentrations()'),
        (parsing._is_save_concentrations, 'saveConcentrations()', 'saveParameters()'),
        (parsing._is_save_parameters, 'saveParameters()', 'saveConcentrations()'),
    ],
)
def test_action_predicates(predicate, matching, non_matching):
    assert predicate(matching) is True
    assert predicate(non_matching) is False
