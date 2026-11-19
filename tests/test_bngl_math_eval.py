"""PyBNF reads BNGL math the way BNG2.pl does, not the way Python's ``eval`` does.

The bngsim bridges compute some BNGL math themselves: the parameters block on the network-free
bridge, quoted action arguments and ``simulate``/``parameter_scan`` settings, and the initial
amounts the ``.net`` bridge re-derives. They used to hand that text to Python's ``eval``, which
reads ``^`` as bitwise XOR. So ``10^5`` was 15, ``10^-12`` was -2, and a network-free fit of a
published model (Hat_2016) ran with 36 wrong parameter values and no error.

The oracle throughout is BNG2.pl 2.9.3 itself:

* ``bngl_math_caret.xml`` is what BNG2.pl wrote for ``bngl_math_caret.bngl``. Its ``value``
  attributes are the numbers BNG2.pl computed for each parameter.
* ``_ACTION_ORACLE`` holds the values BNG2.pl gave quoted action arguments.
  ``test_action_oracle_matches_bng2pl`` re-derives them from a live BNG2.pl where one is
  installed, so the table cannot drift from BioNetGen.
* An unquoted action argument is Perl to BNG2.pl, where ``^`` is XOR (``10^3`` is 9). Nobody
  means that, so PyBNF refuses an unquoted ``^`` or ``**`` when it loads the model.
* A negative number raised to a power is read two ways inside BioNetGen itself.
  ``test_bionetgen_reads_a_minus_before_a_power_two_ways`` shows it: BNG2.pl's arithmetic gives
  ``-2^2`` as 4, but it copies the text into the network file, where the simulators give -4.
  PyBNF cannot know which the author meant, so it refuses the expression and asks for
  parentheses. BNG2.pl brackets every other signed or chained power when it writes the file,
  so those are read as BNG2.pl reads them, except in text taken from a network file, which is
  read by run_network and bngsim with their own grammar.
"""

import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import pybnf.bngsim_model as bngsim_model
from pybnf.bngsim_model import expressions, net_model, parsing
from pybnf.printing import PybnfError
from pybnf.pset import BNGLModel, FailedSimulationError, FreeParameter, ModelError, PSet

from ._bngl_differential import _locate_bng2

FIXTURES = Path(__file__).resolve().parent / 'bngl_files'
CARET_BNGL = FIXTURES / 'bngl_math_caret.bngl'
CARET_XML = FIXTURES / 'bngl_math_caret.xml'
CARET_IC_NET = FIXTURES / 'bngl_math_caret_ic.net'
DECAY_NET = FIXTURES / 'e2e_ode_decay.net'


def _bng2pl_parameter_values(xml_path):
    """The value BNG2.pl computed for each parameter, read from the XML it wrote."""
    root = ET.parse(xml_path).getroot()
    return {
        p.get('id'): float(p.get('value'))
        for p in root.iter()
        if p.tag.endswith('Parameter') and p.get('id') is not None
    }


# --------------------------------------------------------------- the parameters block
def test_network_free_parameter_values_match_bng2pl():
    """Every parameter the network-free bridge pushes into NFsim equals BNG2.pl's value."""
    lines = CARET_BNGL.read_text().splitlines(keepends=True)
    ours = expressions._evaluate_bngl_params(expressions._parse_bngl_param_block(lines))
    theirs = _bng2pl_parameter_values(CARET_XML)

    assert set(ours) == set(theirs)
    for name, value in theirs.items():
        assert ours[name] == pytest.approx(value, rel=1e-7), name


def test_a_fitted_value_flows_through_a_power():
    """``a_ref = 10^p_exp``: a fitted ``p_exp`` must reach ``a_ref`` as a power."""
    lines = CARET_BNGL.read_text().splitlines(keepends=True)
    out = expressions._evaluate_bngl_params(
        expressions._parse_bngl_param_block(lines), {'p_exp': 3.0})
    assert out['p_exp'] == 3.0
    assert out['a_ref'] == pytest.approx(10.0 ** 3)


def test_committed_xml_is_what_bng2pl_writes(tmp_path):
    """The committed oracle is BNG2.pl's current answer, not a stale copy."""
    bng2 = _locate_bng2()
    if bng2 is None:
        pytest.skip('BNG2.pl not found (set BNGPATH)')
    model = tmp_path / CARET_BNGL.name
    model.write_text(CARET_BNGL.read_text())
    subprocess.run(['perl', bng2, model.name], cwd=tmp_path, check=True,
                   capture_output=True, timeout=120)
    assert _bng2pl_parameter_values(tmp_path / CARET_XML.name) == \
        _bng2pl_parameter_values(CARET_XML)


@pytest.mark.bngsim
@pytest.mark.bngsim_nfsim
def test_network_free_simulation_uses_the_bngl_rate(tmp_path):
    """End to end: ``k 10^0`` is a decay rate of 1, which Python read as 10.

    Each of 1000 molecules survives to t = 0.5 with probability exp(-0.5), so the count is
    binomial with mean 606.5 and SD 15.4. At k = 10 the mean would be 6.7.
    """
    action = 'simulate({method=>"nf",t_start=>0,t_end=>0.5,n_steps=>1,suffix=>"tc"})'
    model = bngsim_model.BngsimNfModel(
        'bngl_math_caret', [action], [('simulate', 'tc')], [], str(CARET_XML),
        bngl_model_lines=CARET_BNGL.read_text().splitlines(keepends=True), param_names=())
    finals = []
    for i in range(20):
        model._pybnf_replicate_index = i
        data = model.execute(str(tmp_path), 'rep%d' % i, 60)['tc']
        finals.append(data.data[-1, data.cols['Atot']])

    p = math.exp(-0.5)
    se = math.sqrt(1000 * p * (1 - p) / len(finals))
    assert abs(np.mean(finals) - 1000 * p) < 5 * se


@pytest.mark.bngsim
@pytest.mark.bngsim_nfsim
def test_network_free_set_concentration_evaluates_a_quoted_expression(tmp_path):
    """``setConcentration("A()", "2*10^3")`` sets 2000 molecules. The bridge used to log a
    warning and skip the action, leaving 1000."""
    actions = ['setConcentration("A()", "2*10^3")',
               'simulate({method=>"nf",t_start=>0,t_end=>1e-6,n_steps=>1,suffix=>"tc"})']
    model = bngsim_model.BngsimNfModel(
        'bngl_math_caret', actions, [('simulate', 'tc')], [], str(CARET_XML),
        bngl_model_lines=CARET_BNGL.read_text().splitlines(keepends=True), param_names=())
    data = model.execute(str(tmp_path), 'setconc', 60)['tc']
    assert data.data[0, data.cols['Atot']] == 2000


# ------------------------------------------------------------------ action arguments
# What BNG2.pl 2.9.3 does with each quoted argument: the value it gives, read from the
# network it writes (writeNET) and, for t_end, from the TEND it reports.
_ACTION_ORACLE = {
    'setParameter': [('"10^3"', 1000.0), ('"10^-2"', 0.01), ('"3*10^-4"', 3e-4)],
    'setConcentration': [('"10^3"', 1000.0), ('"2^3^2"', 64.0), ('"2^-3"', 0.125)],
    'addConcentration': [('"10^2"', 100.0)],
}
_T_END_ORACLE = ('"2^3"', 8.0)


@pytest.mark.parametrize('arg, expected', _ACTION_ORACLE['setParameter'])
def test_set_parameter_reads_a_quoted_argument_as_bngl(arg, expected):
    name, value = parsing._parse_set_parameter('setParameter("k", %s)' % arg)
    assert (name, value) == ('k', pytest.approx(expected))


@pytest.mark.parametrize('arg, expected', _ACTION_ORACLE['setConcentration'])
def test_set_concentration_reads_a_quoted_argument_as_bngl(arg, expected):
    line = 'setConcentration("A()", %s)' % arg
    assert parsing._parse_set_concentration(line) == ('A()', pytest.approx(expected))


@pytest.mark.parametrize('arg, expected', _ACTION_ORACLE['addConcentration'])
def test_add_concentration_reads_a_quoted_argument_as_bngl(arg, expected):
    line = 'addConcentration("A()", %s)' % arg
    assert parsing._parse_add_concentration(line) == ('A()', pytest.approx(expected))


def test_a_quoted_expression_over_parameters_is_bngl():
    """``setConcentration("A()", "k0^2")`` against the model's parameters, as the .net bridge
    evaluates it when the action runs."""
    class _Model:
        param_names = ('k0',)

        @staticmethod
        def get_param(name):
            return {'k0': 3.0}[name]

    assert expressions._eval_model_expression('k0^2', _Model()) == 9.0
    assert expressions._eval_model_expression('2*k0^-1', _Model()) == pytest.approx(2.0 / 3.0)


def test_action_oracle_matches_bng2pl(tmp_path):
    """Re-derive ``_ACTION_ORACLE`` from a live BNG2.pl, so the table cannot drift."""
    bng2 = _locate_bng2()
    if bng2 is None:
        pytest.skip('BNG2.pl not found (set BNGPATH)')
    params, species, actions = [], [], []
    for i, (arg, _) in enumerate(_ACTION_ORACLE['setParameter']):
        params.append('p%d 1' % i)
        actions.append('setParameter("p%d", %s)' % (i, arg))
    calls = ([('setConcentration', a, v) for a, v in _ACTION_ORACLE['setConcentration']]
             + [('addConcentration', a, v) for a, v in _ACTION_ORACLE['addConcentration']])
    for i, (call, arg, _) in enumerate(calls):
        species.append('S%d() 0' % i)
        actions.append('%s("S%d()", %s)' % (call, i, arg))
    model = '\n'.join(
        ['begin model', 'begin parameters'] + params + ['end parameters',
         'begin molecule types'] + ['S%d()' % i for i in range(len(calls))]
        + ['end molecule types', 'begin seed species'] + species + ['end seed species',
         'begin reaction rules', 'S0() -> S0() + S1() p0', 'end reaction rules', 'end model',
         'generate_network({overwrite=>1})'] + actions
        + ['writeNET({prefix=>"after"})',
           'simulate({method=>"ode",t_end=>%s,n_steps=>1,prefix=>"tend"})' % _T_END_ORACLE[0]])
    (tmp_path / 'oracle.bngl').write_text(model + '\n')
    log = subprocess.run(['perl', bng2, 'oracle.bngl'], cwd=tmp_path, check=True,
                         capture_output=True, text=True, timeout=120).stdout

    net = (tmp_path / 'after.net').read_text()
    values = dict(re.findall(r'^\s*\d+\s+(\S+)\s+(\S+)\s+#', net, re.M))
    for i, (_, expected) in enumerate(_ACTION_ORACLE['setParameter']):
        assert float(values['p%d' % i]) == pytest.approx(expected)
    amounts = re.findall(r'^\s*\d+\s+S(\d+)\(\)\s+(\S+)\s*$', net, re.M)
    for index, amount in amounts:
        expected = calls[int(index)][2]
        assert float(values.get(amount, amount)) == pytest.approx(expected)
    assert len(amounts) == len(calls)
    assert 'TEND:%g' % _T_END_ORACLE[1] in log
    # And run_network, which simulates the network file BNG2.pl writes for the run, starts
    # from the same amounts: the table is what BioNetGen simulates, not only what it prints.
    start = np.loadtxt(tmp_path / 'tend.cdat', ndmin=2)[0, 1:]
    assert start == pytest.approx([v for _, _, v in calls])


@pytest.mark.bngsim
def test_net_bridge_runs_quoted_bngl_arguments(tmp_path):
    """End to end on the .net bridge: k = "10^-1", S(0) = "10^3", t_end = "2^3". Exponential
    decay then leaves 1000 exp(-0.8) at t = 8. Python's eval gave k = -11, S(0) = 9 and
    t_end = 1."""
    actions = ['setParameter("k", "10^-1")', 'setConcentration("S()", "10^3")',
               'simulate({method=>"ode",t_start=>0,t_end=>%s,n_steps=>4,suffix=>"tc"})'
               % _T_END_ORACLE[0]]
    model = bngsim_model.BngsimModel(
        DECAY_NET.stem, actions, [('simulate', 'tc')], [], nf=str(DECAY_NET))
    model.param_set = PSet([])
    data = model.execute(str(tmp_path), 'quoted', 60)['tc']
    time = data.data[:, data.cols['time']]
    stot = data.data[:, data.cols['Stot']]
    assert time[-1] == pytest.approx(8.0)
    assert stot[0] == pytest.approx(1000.0)
    assert stot[-1] == pytest.approx(1000.0 * math.exp(-0.8), rel=1e-5)


@pytest.mark.bngsim
@pytest.mark.parametrize('carried', [False, True], ids=['from the seed', 'from a carried state'])
def test_a_scan_point_recomputes_a_quoted_set_concentration(tmp_path, carried):
    """A setConcentration expression is re-evaluated at each parameter_scan point, on both scan
    paths: from the seed, and from the state a simulate left (bngsim's on_point hook). Where it
    is undefined (k = 1 here) that point is a failed simulation. The carried-state path used
    to log it and skip it, leaving the species at its carried amount. Either way S starts each
    point at 1/(k-1)+10, so S(1) = (1/(k-1)+10) exp(-k)."""
    def scan(values):
        actions = ['setConcentration("S()", "1/(k-1)+10")']
        if carried:
            actions.append('simulate({method=>"ode",t_end=>0.5,n_steps=>1,suffix=>"pre"})')
        actions.append('parameter_scan({parameter=>"k",par_scan_vals=>%s,t_end=>1,n_steps=>1,'
                       'suffix=>"scan"})' % values)
        suffixes = [('simulate', 'pre')] * carried + [('parameter_scan', 'scan')]
        model = bngsim_model.BngsimModel(
            DECAY_NET.stem, actions, suffixes, [], nf=str(DECAY_NET))
        model.param_set = PSet([])
        return model.execute(str(tmp_path), 'scan', 60)['scan'].data

    data = scan('[0.3,2]')
    for k, s_end in data:
        assert s_end == pytest.approx((1 / (k - 1) + 10) * math.exp(-k), rel=1e-5)
    with pytest.raises(FailedSimulationError, match='parameter_scan point'):
        scan('[0.3,1]')


@pytest.mark.bngsim
@pytest.mark.parametrize('action', ['setConcentration("S", 50)', 'addConcentration("S", 5)'])
def test_a_species_name_bngsim_cannot_find_is_refused_when_the_bridge_is_built(action):
    """BNG2.pl finds ``S`` as the species ``S()``; bngsim needs the exact name. The action used
    to be logged and skipped while running. Refusing it when the bridge is built is where
    bngl_backend = auto falls back to BNG2.pl, which runs it as written."""
    with pytest.raises(PybnfError, match='not in the network'):
        bngsim_model.BngsimModel(DECAY_NET.stem, [action], [], [], nf=str(DECAY_NET))
    bngsim_model.BngsimModel(  # the network's own name is accepted
        DECAY_NET.stem, [action.replace('"S"', '"S()"')], [], [], nf=str(DECAY_NET))


@pytest.mark.parametrize('condition, problem', [
    ('Stot<10', None), ('Stot < 1e1', None), ('time>5', None), ('(Stot-1)/2 >= 0.5', None),
    ('Stot<exp(3.45)', 'not an observable'), ('Stot<S0/3', 'not an observable'),
    ('Stot>1 && Stot<5', 'reads as Python'), ('1<Stot<5', 'chains comparisons'),
])
def test_which_stop_conditions_bngsim_reads_as_run_network_does(condition, problem):
    """bngsim evaluates a stop condition as Python over the observables and time, and passes
    over a row it cannot evaluate. Only what reads the same in run_network's BNGL is let
    through to it."""
    found = net_model._stop_condition_problem(condition, {'Stot', 'time'})
    if problem is None:
        assert found is None
    else:
        assert problem in found


@pytest.mark.bngsim
def test_a_stop_condition_bngsim_cannot_read_is_refused_when_the_bridge_is_built(tmp_path):
    """``Stot<S0/3`` stops BNG2.pl at the first row where 100 exp(-0.3 t) < 100/3, t = 4 here
    by hand (ceil(ln(3) / 0.3)). bngsim never stopped, because S0 is not bound in its eval."""
    action = 'simulate({method=>"ode",t_end=>20,n_steps=>20,stop_if=>"%s",suffix=>"tc"})'
    with pytest.raises(PybnfError, match='S0, which is not an observable'):
        bngsim_model.BngsimModel(DECAY_NET.stem, [action % 'Stot<S0/3'], [('simulate', 'tc')],
                                 [], nf=str(DECAY_NET))
    model = bngsim_model.BngsimModel(
        DECAY_NET.stem, [action % 'Stot<100/3'], [('simulate', 'tc')], [], nf=str(DECAY_NET))
    model.param_set = PSet([])
    data = model.execute(str(tmp_path), 'stop', 60)['tc']
    assert data.data[-1, data.cols['time']] == math.ceil(math.log(3) / 0.3)


# ---------------------------------------------------- initial amounts on the .net bridge
@pytest.mark.bngsim
@pytest.mark.parametrize('r', [2.0, 3.0])
def test_net_bridge_rederives_inline_initial_amounts_as_bngl(r, tmp_path):
    """``A() 10^R`` and ``B() 2^3`` written inline in a network file. The bridge re-derives
    each initial amount from the current parameters, and used to set B to 2 XOR 3 = 1."""
    model = bngsim_model.BngsimModel(
        CARET_IC_NET.stem,
        ['simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>1,suffix=>"tc"})'],
        [('simulate', 'tc')], [], nf=str(CARET_IC_NET))
    model.param_set = PSet([FreeParameter('R', 'uniform_var', 0.0, 10.0, value=r)])
    data = model.execute(str(tmp_path), 'ic', 60)['tc']
    assert data.data[0, data.cols['Atot']] == pytest.approx(10.0 ** r)
    assert data.data[0, data.cols['Btot']] == pytest.approx(8.0)


class _FakeEngineModel:
    """Just enough of a bngsim model for the initial-amount refresh."""

    def __init__(self, params, refuse=()):
        self.params = dict(params)
        self.refuse = set(refuse)
        self.set = {}
        self.param_names = tuple(self.params)

    def get_param(self, name):
        return self.params[name]

    def set_concentration(self, species, value):
        if species in self.refuse:
            raise RuntimeError('no such species')
        self.set[species] = value

    def save_concentrations(self):
        pass


def _refresher(initializers):
    bridge = object.__new__(bngsim_model.BngsimModel)
    bridge.name = 'm'
    bridge._net_species_initializers = list(initializers)
    bridge._sensitivity_request = None
    return bridge


def test_initial_amount_refresh_reads_bngl():
    model = _FakeEngineModel({'R': 3.0})
    _refresher([('A()', '10^R'), ('B()', '2^3')])._sync_species_initial_concentrations(model)
    assert model.set == {'A()': 1000.0, 'B()': 8.0}


@pytest.mark.parametrize('initializers, params, refuse, error', [
    ([('A()', 'not_a_parameter')], {'R': 3.0}, (), PybnfError),
    ([('A()', '-R^2')], {'R': 3.0}, (), PybnfError),
    ([('A()', '2^R^2')], {'R': 3.0}, (), PybnfError),
    ([('A()', '1/(R - 3)')], {'R': 3.0}, (), FailedSimulationError),
    ([('A()', 'R')], {'R': 3.0}, ('A()',), PybnfError),
], ids=['unknown name', 'minus before a power', 'chained powers', 'undefined at this point',
        'not accepted'])
def test_initial_amount_refresh_fails_loudly(initializers, params, refuse, error):
    """An initial amount that could not be re-derived or set used to be logged at debug level
    and skipped, so the species kept its value from the previous parameter set. A minus before
    a power and a chain of powers are refused because the file's own readers, run_network and
    bngsim, read them differently from BNGL (bngsim gives 2^3^2 as 512)."""
    model = _FakeEngineModel(params, refuse)
    with pytest.raises(error):
        _refresher(initializers)._sync_species_initial_concentrations(model)


# ------------------------------------------ a minus sign directly before a power
def test_bionetgen_reads_a_minus_before_a_power_two_ways(tmp_path):
    """Why PyBNF refuses ``-2^2`` where it evaluates BNGL itself: BioNetGen has two answers.

    BNG2.pl's own arithmetic (its XML ``value``) gives (-2)^2 = 4. The network file it writes
    for a run keeps the text ``-2^2``, and run_network, which simulates that file, starts the
    species at -(2^2) = -4.
    """
    bng2 = _locate_bng2()
    if bng2 is None:
        pytest.skip('BNG2.pl not found (set BNGPATH)')
    (tmp_path / 'minus.bngl').write_text(
        'begin model\nbegin parameters\n  a -2^2\n  k 0\nend parameters\n'
        'begin molecule types\n  A()\nend molecule types\n'
        'begin seed species\n  A() a\nend seed species\n'
        'begin reaction rules\n  A() -> A() + A() k\nend reaction rules\nend model\n'
        'generate_network({overwrite=>1})\nwriteXML()\n'
        'simulate({method=>"ode",t_end=>1,n_steps=>1})\n')
    subprocess.run(['perl', bng2, 'minus.bngl'], cwd=tmp_path, check=True,
                   capture_output=True, timeout=120)
    assert _bng2pl_parameter_values(tmp_path / 'minus.xml')['a'] == 4.0
    assert np.loadtxt(tmp_path / 'minus.cdat', ndmin=2)[0, 1] == -4.0


@pytest.mark.parametrize('text', ['"-2^2"', '2*-3^2', '-2.5^2', 'exp(-1^2)'])
def test_a_negative_number_raised_to_a_power_is_refused(text):
    """The one form BNG2.pl copies into the network file unchanged, where it reads -4."""
    with pytest.raises(PybnfError, match='parentheses'):
        expressions._eval_numeric(text)


@pytest.mark.parametrize('text, expected', [
    ('-k^2', 9.0), ('-(2)^2', 4.0), ('-exp(0)^2', 1.0), ('10^-2^2', 1e-4), ('2^3^2', 64.0),
])
def test_signs_and_chains_bng2pl_brackets_are_read_as_bng2pl_reads_them(text, expected):
    """BNG2.pl writes these into the network file with parentheses, as (-k)^2, (-2)^2,
    (-exp(0))^2, (10^-2)^2 and (2^3)^2 (checked with BNG2.pl 2.9.3), so every reader agrees."""
    assert expressions._eval_numeric(text, {'k': 3.0}) == pytest.approx(expected)


def test_a_minus_sign_before_a_power_in_the_parameters_block_ends_the_run():
    """In the network-free parameters block the refusal is a :class:`PybnfError`, which ends
    the run with the reason. A plain error there was scored as one failed simulation per
    evaluation, and the fit carried on with nothing but failures."""
    with pytest.raises(PybnfError, match='parentheses'):
        expressions._evaluate_bngl_params([('k', '3'), ('a', '-2^2')])


def test_an_undefined_value_is_one_failed_simulation():
    """A value undefined only at this parameter set (here 1/(k - 3) at k = 3) is a failed
    simulation, not a broken model: the next parameter set may be fine."""
    with pytest.raises(FailedSimulationError):
        expressions._evaluate_bngl_params([('k', '1'), ('a', '1/(k - 3)')], {'k': 3.0})
    assert expressions._evaluate_bngl_params(
        [('k', '1'), ('a', '1/(k - 3)')], {'k': 4.0})['a'] == 1.0


@pytest.mark.parametrize('text, expected', [
    ('(-2)^2', 4.0), ('-(2^2)', -4.0), ('10^-2', 0.01), ('3-2^2', -1.0), ('-2*3', -6.0),
])
def test_unambiguous_signs_are_accepted(text, expected):
    assert expressions._eval_numeric(text) == pytest.approx(expected)


# ------------------------------------------------- unquoted ^ and ** refused at load
_MODEL_HEAD = """begin model
begin parameters
  k 1
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 1
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A() -> 0 k
end reaction rules
end model
"""


def _load(tmp_path, actions):
    path = tmp_path / 'm.bngl'
    path.write_text(_MODEL_HEAD + actions + '\n')
    return BNGLModel(str(path), suppress_free_param_error=True)


@pytest.mark.parametrize('action', [
    'setConcentration("A()", 10^3)',
    'setParameter("k", 10^-2)',
    'simulate({method=>"ode",t_end=>10^1,n_steps=>2})',
    'simulate({method=>"ode",t_end=>2**3,n_steps=>2})',
    'generate_network({overwrite=>1,max_iter=>2^4})',
    'simulate({method=>"ode",t_end=>log(100),n_steps=>2})',
    'setConcentration("A()", 2*exp(1))',
    'simulate({method=>"ode",t_end=>8,n_steps=>"2^3"})',
    'simulate({method=>"ode",t_end=>8,n_steps=>2,atol=>"10^-10"})',
    'parameter_scan({parameter=>"k",par_min=>"10^-2",par_max=>1,n_scan_pts=>2,t_end=>1})',
    'setConcentration("A()", "2**3")',
    'simulate({method=>"ode",t_end=>8,n_steps=>2,stop_if=>"Atot<2^5"})',
])
def test_action_math_bng2pl_does_not_read_as_bngl_is_refused(tmp_path, action):
    """BNG2.pl runs an unquoted argument as Perl, where 10^3 is 9 (bitwise XOR) and log is
    Perl's. A quoted setting other than t_start and t_end goes to run_network as text, which
    reads "10^-10" as 10. And run_network and bngsim reject **. Every backend loads the model
    through BNGLModel, so the refusal covers BNG2.pl and bngsim alike."""
    with pytest.raises(ModelError) as info:
        _load(tmp_path, 'begin actions\n' + action + '\nend actions')
    message = str(info.value)
    assert 'line 19' in message
    assert action in message
    assert 'quote' in message.lower()


def test_an_unquoted_power_in_a_protocol_block_is_refused(tmp_path):
    with pytest.raises(ModelError, match='line 20'):
        _load(tmp_path, 'begin protocol\n'
                        'simulate({method=>"ode",t_end=>1,n_steps=>1})\n'
                        'setConcentration("A()", 10^3)\n'
                        'end protocol')


@pytest.mark.parametrize('actions', [
    'begin actions\nsetConcentration("A()", "10^3")\nend actions',
    "begin actions\nsetParameter('k', '2^3')\nend actions",
    "begin actions\naddConcentration('A()', '10^2')\nend actions",
    'begin actions\nsimulate({method=>"ode",t_end=>"2^3",n_steps=>2}) # 2^3 is 8\nend actions',
    'begin actions\nparameter_scan({parameter=>"k",par_min=>1,par_max=>2,n_scan_pts=>2,'
    't_start=>"2^0",t_end=>"2^3"})\nend actions',
    'simulate({method=>"ode",t_end=>10*60,n_steps=>2})',
    'begin actions\nsimulate({method=>"ode","t_end"=>"2^3",n_steps=>2,suffix=>"x^2"})\n'
    'saveConcentrations("pre^wash")\nend actions',
])
def test_a_quoted_power_or_a_comment_is_accepted(tmp_path, actions):
    _load(tmp_path, actions)


# ------------------------------------------- what BNG2.pl rejects is rejected, not guessed
def test_a_double_star_is_a_power():
    """BNG2.pl's arithmetic reads ``**`` as ``^``. (run_network and bngsim reject it in a network
    file, and BNGLModel refuses it in an action, so it reaches the bridges only from a
    parameters block, as BNG2.pl's XML value.)"""
    assert expressions._eval_numeric('"2**3"') == 8.0
    assert expressions._evaluate_bngl_params([('a', '2**3'), ('b', 'a**2')]) == {'a': 8.0, 'b': 64.0}


@pytest.mark.parametrize('text', ['log(10)', 'pow(2, 10)', 'floor(2.5)', 'pi'])
def test_names_bngl_does_not_have_are_errors(text):
    """Python's eval accepted these. BNGL has no ``log``, ``pow`` or ``floor`` function and no
    ``pi`` constant (its natural log is ``ln`` and its constant is ``_pi()``), and BNG2.pl
    rejects them, so they are errors rather than plausible numbers. A PybnfError ends the run:
    every job would hit it."""
    with pytest.raises(PybnfError):
        expressions._eval_numeric(text)
