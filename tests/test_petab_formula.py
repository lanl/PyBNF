"""Tests for the PEtab ``observableFormula`` expression layer (#407, ADR-0035/0036).

ADR-0036 makes a PEtab ``observableFormula`` a **measurement model** -- evaluated as a
post-simulation transform over the output trajectory (the observation layer in
:mod:`pybnf.measurement`), never by editing the model file. This supersedes ADR-0035's
``begin functions`` *synthesis into the model*; the reversible translator's surviving
production directions are:

* :func:`~pybnf.petab.formula.bngl_body_to_petab_math` -- the exporter's inlining mode (a BNGL
  function body -> a PEtab math ``observableFormula``), which still generates the round-trip
  oracle; its precedence/spelling logic is graded here.
* :func:`~pybnf.petab.formula.compile_petab_formula` -- the layer's compiler (PEtab math -> a
  numpy callable), graded by numeric hand-computation in ``tests/test_measurement_layer.py``.

The oracles, weakest to strongest:

1. **Export-inline translator unit tests** -- ``bngl_body_to_petab_math`` on crafted bodies:
   the ``^``/``ln``/``log10``/``sqrt`` spellings, the ``sqrt`` precedence defect, the ``func()``
   reference convention, the standing self-check, and free-symbol validation. The body is
   BNGL, so its oracle is BioNetGen itself (#908): a table of the values BNG2.pl computes for
   bodies that exercise every place BNGL's grammar differs from PEtab's, re-derived from a
   live BNG2.pl when one is on PATH.
2. **Syntactic round trip (fast tier).** A crafted BNGL model whose measurement model is a
   multi-operator function exports *with inlining* -> imports (to a conf measurement-model
   line, carrying the model verbatim) -> re-exports; the ``observableFormula`` is graded
   sympy-equal across the hop, and the imported job carries a measurement model (not a
   synthesized function).
3. **Semantic round trip (``-m recovery``, bngsim).** The imported job is simulated through the
   real bngsim backend and the measurement layer's computed column reproduces the original
   model function's trace -- catching a self-consistent-but-wrong translation a syntactic
   oracle would miss.

``petab``/``sympy`` is the optional ``pybnf[petab]`` extra; the expression-path tests
``importorskip('petab')``. The petab-absent contract (a pointed "install pybnf[petab]" error,
not an ``ImportError``) is tested *without* skipping -- the bare-name path's dependency-free
guarantee.
"""

import builtins
import re
import sys

import numpy as np
import pytest

from pybnf.petab import export_job, import_job
from pybnf.petab._bngl import parse_model
from pybnf.petab.formula import bngl_body_to_petab_math
from pybnf.printing import PybnfError

# A crafted BNGL model whose measurement model is a Boehm-style quotient of sums over
# observables (obsA/obsB) and parameters (kA/kB/kC) -- a multi-operator expression with
# no SBML required. Actionless (new-era, ADR-0028): export reads the experiment surface
# and the recovery sim synthesizes the action from the data's time column.
CRAFTED_MODEL = """\
begin model
  begin parameters
    kA 2
    kB 3
    kC 0.5
  end parameters
  begin molecule types
    A()
    B()
  end molecule types
  begin seed species
    A() 10
    B() 4
  end seed species
  begin observables
    Molecules obsA A()
    Molecules obsB B()
  end observables
  begin functions
    pRel() = (100*obsA + 200*obsB*kA)/(obsB + kB*obsA + 2*kC*obsB)
  end functions
  begin reaction rules
    A() -> B() kA
  end reaction rules
end model
"""

CRAFTED_EXP = '# time\tpRel\n0\t5\n1\t6\n2\t7\n'

CRAFTED_CONF = (
    'edition = 2\njob_type = de\nobjective = sos\nmodel: crafted.bngl\n'
    'experiment: meas, data: meas.exp\n'
    'uniform_var = kA 0 10\nuniform_var = kB 0 10\nuniform_var = kC 0 10\n')


def _entities(model_text=CRAFTED_MODEL):
    return parse_model(model_text)


def _sympy_equal(petab_expr_a, petab_expr_b):
    """True iff two PEtab math strings denote the same function.

    By numeric sampling at distinct positive points, not symbolic ``simplify``: petab
    floatifies literals (a ``sqrt`` parses back with a ``1.0/2.0`` Float exponent, not an
    exact ``Rational(1/2)``), and sympy treats Float-vs-exact powers as unequal under
    ``simplify`` -- so a symbolic test false-rejects a correct ``sqrt`` translation.
    Positive points keep ``sqrt``/``log`` real; multiple points rule out coincidental
    agreement (the corrupt ``z/2`` and ``sqrt(z)`` collide only at ``z=4``).
    """
    import sympy as sp
    from petab.v2.math import sympify_petab
    ea = sympify_petab(petab_expr_a, evaluate=False)
    eb = sympify_petab(petab_expr_b, evaluate=False)
    syms = sorted(ea.free_symbols | eb.free_symbols, key=str)
    for k in range(1, 6):
        subs = {s: sp.Rational(3 + 2 * k + 5 * i, 7) for i, s in enumerate(syms)}
        va, vb = float(sp.N(ea.subs(subs))), float(sp.N(eb.subs(subs)))
        if abs(va - vb) > 1e-7 * max(1.0, abs(vb)):
            return False
    return True


# ---------------------------------------------------------------------------
# 1. Export-inline translator (BNGL function body -> PEtab math observableFormula)
# ---------------------------------------------------------------------------

class TestExportInlineTranslator:

    @pytest.mark.parametrize('body', [
        'kA*obsA + kB',
        '(100*obsA + 200*obsB*kA)/(obsB + kB*obsA + 2*kC*obsB)',   # quotient of sums
        'obsA^2 + 2*obsA - kC',
        'kA*(obsA - obsB)/(kB + kC)',
        'sqrt(obsA)',                                              # the petab ^1/2 defect
        'kA*sqrt(obsA) + kB',
        '(obsA + obsB)/sqrt(kC)',
        'sqrt(kA*obsA + kB)',
    ])
    def test_inlined_formula_is_equivalent_to_the_body(self, body):
        # The emitted observableFormula denotes the same function as the BNGL body. The
        # forward translator's own _assert_round_trips guards this internally (it re-parses
        # its output); here we confirm it externally against the body too. All these bodies
        # are already valid PEtab math, so _sympy_equal can parse both sides.
        pytest.importorskip('petab')
        out = bngl_body_to_petab_math(body, _entities())
        assert _sympy_equal(out, body)

    def test_sqrt_serializes_precedence_safe_not_the_petab_defect(self):
        # Guards the petab 0.8.x petab_math_str defect (ADR-0035): a sqrt must NOT export as
        # the unparenthesized `x ^ 1/2` (which re-parses as x/2 and silently corrupts the
        # measurement model). Our printer parenthesizes the half-power; the emitted string
        # must denote sqrt, not x/2, and be valid PEtab math the validator re-parses.
        pytest.importorskip('petab')
        out = bngl_body_to_petab_math('sqrt(obsA)', _entities())
        assert ' ^ 1/2' not in out                    # not the defective form
        assert _sympy_equal(out, 'sqrt(obsA)')         # means sqrt...
        assert not _sympy_equal(out, 'obsA/2')         # ...not the corruption

    @pytest.mark.parametrize('body, defective', [
        ('sqrt(obsA)', 'obsA ^ 1/2'),              # petab 0.8's buggy sqrt serialization
        ('-kC^2 + obsA', '-kC ^ 2.0 + obsA'),      # what the PEtab-grammar reading printed
        ('kA^kB^kC + obsA', 'kA ^ (kB ^ kC) + obsA'),   # (#908)
    ])
    def test_guard_refuses_a_formula_that_disagrees_with_the_body(self, monkeypatch, body,
                                                                  defective):
        # The standing tripwire: if the printer ever emits a formula whose PEtab value is not
        # BioNetGen's value of the body, refuse it loudly rather than corrupt silently. The
        # guard this replaced compared PEtab's reading of the body with PEtab's reading of the
        # output, so the second and third strings -- exactly what #908 reported being emitted
        # -- passed it.
        pytest.importorskip('petab')
        import pybnf.petab._bngl_math as M
        import pybnf.petab.formula as F
        monkeypatch.setattr(M, 'to_petab', lambda tree: defective)
        with pytest.raises(PybnfError, match='silently change the measurement model'):
            F.bngl_body_to_petab_math(body, _entities())

    def test_guard_refuses_a_body_it_cannot_check(self):
        # A body that is undefined everywhere (a division by zero) leaves nothing to check the
        # formula against, so no formula is emitted.
        pytest.importorskip('petab')
        with pytest.raises(PybnfError, match='Could not check.*undefined'):
            bngl_body_to_petab_math('obsA/(kA - kA)', _entities())

    def test_function_reference_strips_parens_on_the_petab_side(self):
        # A body referencing another global function writes it `f()`; PEtab math has no
        # zero-arg user function, so the inlined formula references it as a bare symbol.
        pytest.importorskip('petab')
        ent = parse_model(
            'begin observables\n Molecules obsA A()\nend observables\n'
            'begin functions\n g() = obsA*2\n h() = g()^2 + obsA\nend functions\n')
        petab_math = bngl_body_to_petab_math('g()^2 + obsA', ent)
        assert 'g(' not in petab_math            # PEtab has no zero-arg user function
        assert _sympy_equal(petab_math, 'g^2 + obsA')

    def test_unknown_symbol_is_an_error_not_a_free_parameter(self):
        pytest.importorskip('petab')
        with pytest.raises(PybnfError, match='not a parameter, observable, or function'):
            bngl_body_to_petab_math('obsA + nope', _entities())

    def test_per_measurement_placeholder_is_deferred(self):
        pytest.importorskip('petab')
        with pytest.raises(NotImplementedError, match='placeholder'):
            bngl_body_to_petab_math('obsA * observableParameter1_x', _entities())

    def test_expression_without_petab_raises_pointed_error(self, monkeypatch):
        # The dependency-free guarantee: with petab absent the expression path raises a
        # pointed PybnfError naming the extra, NOT a bare ImportError from the call stack.
        # (No importorskip -- this is exactly the petab-absent contract.)
        real_import = builtins.__import__

        def _block(name, *args, **kwargs):
            if name == 'petab' or name.startswith('petab.'):
                raise ImportError('petab blocked for this test')
            return real_import(name, *args, **kwargs)

        for mod in [m for m in sys.modules if m.startswith('petab')]:
            monkeypatch.delitem(sys.modules, mod, raising=False)
        monkeypatch.setattr(builtins, '__import__', _block)
        with pytest.raises(PybnfError, match=r'pybnf\[petab\]'):
            bngl_body_to_petab_math('obsA + 1', _entities())


# ---------------------------------------------------------------------------
# 1b. The body is BNGL, so it is read with BioNetGen's grammar, not PEtab's (#908)
# ---------------------------------------------------------------------------
#
# The oracle is BioNetGen. BNG_ORACLE_MODEL, with every function of BNG_ORACLE and
# BNG_ORACLE_SIMULATOR_ONLY in its functions block, was run through BNG2.pl 2.9.3 as
# ``BNG2.pl oracle.bngl``; its actions are
#     generate_network({overwrite=>1})
#     writeNET({evaluate_expressions=>1})
#     simulate({method=>"ode",t_end=>1,n_steps=>1,print_functions=>1})
# and the numbers below are the function columns of the .gdat at t = 0 and t = 1, where obsA
# is 10 and 9.048374519482. bngsim 0.15.1 reading the same .net computes the same numbers.
# test_bng_oracle_table_matches_a_live_bng2pl re-derives the table when BNG2.pl is on PATH.

BNG_ORACLE_MODEL = """\
begin model
begin parameters
  kA 2
  kB 3
  kC 2
  kD 0.7
  kE 1.3
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 10
end seed species
begin observables
  Molecules obsA A()
end observables
begin functions
{functions}
end functions
begin reaction rules
  A() -> 0 0.1
end reaction rules
end model
"""
BNG_ORACLE_ACTIONS = """\
generate_network({overwrite=>1})
writeNET({evaluate_expressions=>1})
simulate({method=>"ode",t_end=>1,n_steps=>1,print_functions=>1})
"""
BNG_ORACLE_PARAMS = {'kA': 2.0, 'kB': 3.0, 'kC': 2.0, 'kD': 0.7, 'kE': 1.3}
BNG_ORACLE_OBS_A = (10.0, 9.048374519482)

#: ``(function, body, value at t=0, value at t=1)`` as BNG2.pl computes them.
BNG_ORACLE = [
    # unary minus binds tighter than ^, and ^ is left associative (the issue's f1-f3)
    ('p1', '-kC^2 + obsA', 14.0, 13.04837451948),
    ('p2', 'kA^kB^kC + obsA', 74.0, 73.04837451948),
    ('p3', 'obsA*exp(-(kC/3)^2)', 15.59623497607, 14.11205751573),
    ('p4', 'kA^-kD^kC', 0.3789291416276, 0.3789291416276),
    ('p5', '-exp(kD)^2', 4.055199966845, 4.055199966845),
    ('p6', 'kA - -kB', 5.0, 5.0),
    ('p7', 'kA*-kB^2', 18.0, 18.0),
    ('p8', '+kA^2', 4.0, 4.0),
    ('p9', '2^-1^2', 0.25, 0.25),
    ('p10', 'kA^kB/kC^kD', 4.92457765338, 4.92457765338),
    ('p11', '-(kA)^2', 4.0, 4.0),
    ('p12', '-(2)^2 + obsA', 14.0, 13.04837451948),
    ('p13', 'kA^-2', 0.25, 0.25),
    # literals PEtab's lexer does not accept as written (.5, 5.)
    ('p14', '1e-3*obsA + .5 + 5. + 1E3 + 2.5e+1', 1030.51, 1030.509048375),
    ('p15', 'kA - -2', 4.0, 4.0),
    # a comparison or a logical operator is the number 1 or 0; if() selects on it (an if()
    # whose condition is anything else is refused, see r3)
    ('c1', '(obsA > 9.5)*kA + (obsA < 9.5)*kB', 2.0, 3.0),
    ('c2', 'if(obsA > 9.5, kA, kB)', 2.0, 3.0),
    ('c3', 'if(kA - 2 != 0, 1, 7)', 7.0, 7.0),
    ('c4', 'kA > 1 && kB < 2 || kC == 2', 1.0, 1.0),
    ('c5', 'kA < kB < kC', 1.0, 1.0),
    ('c6', 'obsA >= 10 + kA', 0.0, 0.0),
    ('c7', 'kA != 2 || kB <= 3', 1.0, 1.0),
    ('c8', 'if(kD != 0, kA^2, 0)', 4.0, 4.0),
    ('c9', '(kA == 2) + (kB != 3)', 1.0, 1.0),
    ('c10', 'kA && 0 || kE', 1.0, 1.0),
    # built-in functions and constants, and references to other functions and observables
    ('b1', 'min(kA, kB, kD) + max(kA, kB, kE)', 3.7, 3.7),
    ('b2', 'sum(kA, kB, kC) + avg(kA, kB)', 9.5, 9.5),
    ('b3', '_pi() + _e()', 5.859874482049, 5.859874482049),
    ('b4', 'ln(obsA) + log10(obsA) + log2(obsA) + sqrt(obsA) + abs(-obsA)',
     19.78679084805, 18.39324047807),
    ('b5', 'sin(kD) + cos(kD) + tan(kD) + asin(kD/2) + acos(kD/2) + atan(kD)',
     4.432870546169, 4.432870546169),
    ('b6', 'sinh(kD) + cosh(kD) + tanh(kD) + asinh(kD) + acosh(kE) + atanh(kD/2)',
     4.392663715798, 4.392663715798),
    ('b7', 'min(kA) + max(kE)', 3.3, 3.3),
    ('b8', 'b1()^2 - b3()', 7.830125517951, 7.830125517951),
    ('b9', '-b1()^2', 13.69, 13.69),
    ('b10', 'obsA()', 10.0, 9.048374519482),
    # a negative constant raised to a parameter, which the network file brackets, so every
    # simulator reads it alike (n2 is the form the refusal of '(-2)^x' recommends); and a
    # fractional power of a quotient of several symbols (a Hill-type inverse, as in
    # BNGL-Models' ATG_model_v12). The guard's random points left each without enough
    # points where the body is defined, so all four were refused (review of #908).
    ('n1', '-(2)^kC', 4.0, 4.0),
    ('n2', '(-(2))^kC + obsA', 14.0, 13.04837451948),
    ('n3', '(0-2)^kC', 4.0, 4.0),
    ('n4', '(kA^kE*((kD*(obsA+kB)/(obsA+kB+kC))/(kE-kD*(obsA+kB)/(obsA+kB+kC))))^(1/kE)',
     1.804765605123, 1.777822967906),
]

#: Bodies the simulators evaluate differently from BNG2.pl's own parser, or from one another,
#: which the exporter therefore refuses: the network file carries ``-2^2`` (parentheses and
#: all are dropped), which run_network and bngsim read as -(2^2), while BNG2.pl's parser
#: means (-2)^2 = 4; and run_network's if() takes the first branch only when the condition
#: exceeds 0.5 (Network3's If()), where bngsim, NFsim and BNG2.pl's parser take it whenever
#: the condition is nonzero, so r3 is kB = 3 here and kA = 2 in bngsim.
BNG_ORACLE_SIMULATOR_ONLY = [
    ('r1', '-2^2 + obsA', 6.0, 5.048374519482),
    ('r2', '(-2)^2 + obsA', 6.0, 5.048374519482),
    ('r3', 'if(kD - 0.5, kA, kB)', 3.0, 3.0),
]


def _oracle_model_text():
    rows = BNG_ORACLE + BNG_ORACLE_SIMULATOR_ONLY
    return BNG_ORACLE_MODEL.format(
        functions='\n'.join(f'  {name}() = {body}' for name, body, _v0, _v1 in rows))


def _oracle_values(t_index):
    """Every symbol an oracle formula can name, at the oracle's t = 0 or t = 1."""
    values = dict(BNG_ORACLE_PARAMS, obsA=BNG_ORACLE_OBS_A[t_index])
    values.update({row[0]: row[2 + t_index] for row in BNG_ORACLE})
    return values


def _petab_value(formula, values):
    """The value of a PEtab math formula, read by petab's own parser (the external reader)."""
    from petab.v2.math import sympify_petab
    expr = sympify_petab(formula)
    return float(expr.subs({s: values[str(s)] for s in expr.free_symbols}))


class TestBioNetGenGrammar:
    """``bngl_body_to_petab_math`` reads a body the way BioNetGen does (#908)."""

    @pytest.mark.parametrize('name, body, v0, v1', BNG_ORACLE,
                             ids=[row[0] for row in BNG_ORACLE])
    def test_formula_has_bionetgens_value(self, name, body, v0, v1):
        # Independent oracle: the numbers BNG2.pl's simulation computed for the body, compared
        # with petab's own reading of the emitted formula. The first three rows are the
        # issue's reproduction; on main they came out 6, 522 and 6.412 at t = 0.
        pytest.importorskip('petab')
        ent = parse_model(_oracle_model_text())
        formula = bngl_body_to_petab_math(body, ent, function_name=name)
        for t_index, expected in enumerate((v0, v1)):
            assert _petab_value(formula, _oracle_values(t_index)) == pytest.approx(
                expected, rel=1e-10, abs=1e-12), (formula, t_index)

    @pytest.mark.parametrize('name, body, v0, v1', [
        row if row[0] != 'b4' else pytest.param(*row, marks=pytest.mark.xfail(
            strict=True, raises=TypeError,
            reason="the measurement layer's numpy lambdify writes petab's log10(x) (sympy's "
                   "two-argument log) as numpy.log(x, 10), which fails; reported separately"))
        for row in BNG_ORACLE], ids=[row[0] for row in BNG_ORACLE])
    def test_measurement_layer_evaluates_the_formula_to_bionetgens_value(
            self, name, body, v0, v1):
        # The importer's side of the same contract: PyBNF's own measurement layer (what an
        # imported `observable: <id>, formula: <expr>` line runs) evaluates the emitted formula
        # over BNG2.pl's trajectory and must reproduce BNG2.pl's function column.
        pytest.importorskip('petab')
        from pybnf.data import Data
        from pybnf.measurement import MeasurementModel
        ent = parse_model(_oracle_model_text())
        formula = bngl_body_to_petab_math(body, ent, function_name=name)
        functions = [row[0] for row in BNG_ORACLE]
        headers = ['time', 'obsA'] + functions
        rows = [[float(t), BNG_ORACLE_OBS_A[t]] + [_oracle_values(t)[f] for f in functions]
                for t in (0, 1)]
        trace = Data.from_columns(np.array(rows), headers)
        allowed = set(BNG_ORACLE_PARAMS) | {'obsA'} | set(functions)
        got = MeasurementModel(f'func_{name}', formula, allowed).materialize(
            trace, BNG_ORACLE_PARAMS)
        np.testing.assert_allclose(got, [v0, v1], rtol=1e-10, atol=1e-12)

    def test_issue_reproduction_exports_and_imports_bionetgens_values(self, tmp_path):
        # The issue's own reproduction, end to end: export_job with inline_functions, then
        # import_job, and read both the observables table and the imported conf's formula
        # lines back with petab's parser. BNG2.pl's values at t = 0 are 14, 74 and 15.596.
        pytest.importorskip('petab')
        import csv
        (tmp_path / 'prec.bngl').write_text(PREC_MODEL)
        (tmp_path / 'meas.exp').write_text('# time\tf1\tf2\tf3\n0\t14\t74\t15.596\n'
                                           '1\t13\t73\t14.1\n')
        (tmp_path / 'job.conf').write_text(
            'edition = 2\njob_type = de\nobjective = sos\nmodel: prec.bngl\n'
            'experiment: meas, data: meas.exp\nuniform_var = kA 1 3\n'
            'uniform_var = kB 1 4\nuniform_var = kC 1 3\n')
        export_job(tmp_path / 'job.conf', tmp_path / 'out', inline_functions=True)
        with open(tmp_path / 'out' / 'observables.tsv') as fh:
            exported = {r['observableId']: r['observableFormula']
                        for r in csv.DictReader(fh, delimiter='\t')}
        import_job(tmp_path / 'out' / 'problem.yaml', tmp_path / 'imp')
        from pybnf.parse import ploop
        conf = ploop((tmp_path / 'imp' / 'imported.conf').read_text()
                     .splitlines(keepends=True))
        imported = {k[1]: v for k, v in conf.items()
                    if isinstance(k, tuple) and k[0] == 'measurement'}
        values = {'kA': 2.0, 'kB': 3.0, 'kC': 2.0, 'obsA': 10.0}
        expected = {'func_f1': 14.0, 'func_f2': 74.0, 'func_f3': 15.59623497607}
        for oid, value in expected.items():
            assert _petab_value(exported[oid], values) == pytest.approx(value, rel=1e-10)
            assert _petab_value(imported[oid], values) == pytest.approx(value, rel=1e-10)

    @pytest.mark.parametrize('body, error, match', [
        # the value depends on which part of BioNetGen computes it
        ('-2^2 + obsA', NotImplementedError, 'negative number to a power'),
        ('(-2)^kA', NotImplementedError, 'negative number to a power'),
        ('kA*-3^2', NotImplementedError, 'negative number to a power'),
        ('if(kD - 0.5, kA, kB)', NotImplementedError, r'if\(\) condition.*exceeds 0\.5'),
        ('if(obsA, kA, kB) + 1', NotImplementedError, r'if\(\) condition.*exceeds 0\.5'),
        ('if(-(kA > 1), kA, kB)', NotImplementedError, r'if\(\) condition.*exceeds 0\.5'),
        # valid BNGL with no exact PEtab reading, or none PyBNF can import back
        ('rint(kE) + obsA', NotImplementedError, r'rint\(\) rounds.*no rounding function'),
        ('time()*kA', NotImplementedError, r"time\(\).*measurement layer cannot evaluate"),
        ('mratio(kA, kB, kC)', NotImplementedError, r'mratio\(\)'),
        ("TFUN(obsA, 'curve.tfun')", NotImplementedError, r'TFUN\(\) reads a data file'),
        ('g(obsA)', NotImplementedError, r'calls g\(\) with arguments'),
        # accepted by BNG2.pl, refused by every simulator, so the function has no value
        ('kA**kB', PybnfError, r"'\*\*'.*run_network, NFsim and bngsim"),
        ('!(obsA > 5)', PybnfError, r"'!'.*run_network, NFsim and bngsim"),
        ('~kA', PybnfError, r"'~'.*run_network, NFsim and bngsim"),
        ('obsA ~= 10', PybnfError, r"'~='.*run_network, NFsim and bngsim"),
        # malformed
        ('foo(kA)', PybnfError, r'foo\(\) is not a BioNetGen built-in'),
        ('kA()', PybnfError, r'kA\(\) is not a BioNetGen built-in'),
        ('exp(kA, kB)', PybnfError, r'exp\(\) takes 1 argument'),
        ('--kA', PybnfError, "unexpected '-' where an operand was expected"),
        ('kA +', PybnfError, 'ends where an operand was expected'),
        ('1d3*kA', PybnfError, "unexpected 'd3'"),
    ])
    def test_a_body_without_an_exact_reading_is_refused(self, body, error, match):
        # Every refusal names the function and says what about the body is the problem.
        pytest.importorskip('petab')
        ent = parse_model(_oracle_model_text() + 'begin functions\n g(x) = x\nend functions\n')
        with pytest.raises(error, match=match) as excinfo:
            bngl_body_to_petab_math(body, ent, function_name='f9', model_file='m.bngl')
        assert "BNGL function 'f9' in model 'm.bngl'" in str(excinfo.value)

    def test_export_job_refusal_names_the_column_and_the_model(self, tmp_path):
        pytest.importorskip('petab')
        (tmp_path / 'prec.bngl').write_text(PREC_MODEL.replace(
            'f1() = -kC^2 + obsA', 'f1() = (-2)^kC + obsA'))
        (tmp_path / 'meas.exp').write_text('# time\tf1\n0\t14\n1\t13\n')
        (tmp_path / 'job.conf').write_text(
            'edition = 2\njob_type = de\nobjective = sos\nmodel: prec.bngl\n'
            'experiment: meas, data: meas.exp\nuniform_var = kC 1 3\n')
        with pytest.raises(NotImplementedError,
                           match=r"BNGL function 'f1' in model 'prec.bngl'.*negative number"):
            export_job(tmp_path / 'job.conf', tmp_path / 'out', inline_functions=True)
        # Without inlining the function is referenced by name and nothing is refused.
        export_job(tmp_path / 'job.conf', tmp_path / 'bare')

    def test_the_forms_the_negative_base_refusal_suggests_are_exported(self):
        # The refusal of '(-2)^x' tells the user to write '-(2^x)' or '(-(2))^x'. Following
        # that advice must work: both forms export, to the values of the two readings worked
        # by hand with kC = 2 (the second was refused by the guard, review of #908).
        pytest.importorskip('petab')
        ent = parse_model(_oracle_model_text())
        with pytest.raises(NotImplementedError, match='negative number to a power') as excinfo:
            bngl_body_to_petab_math('(-2)^kC', ent)
        for form, expected in (('-(2^x)', -(2.0 ** 2)), ('(-(2))^x', (-2.0) ** 2)):
            assert f"'{form}'" in str(excinfo.value)
            formula = bngl_body_to_petab_math(form.replace('x', 'kC'), ent)
            assert _petab_value(formula, BNG_ORACLE_PARAMS) == expected, formula

    @pytest.mark.bionetgen
    def test_simulators_disagree_on_an_if_whose_condition_is_not_a_comparison(self, tmp_path):
        """Why if(c, a, b) is refused unless c is a comparison or a logical expression: BNG2.pl's
        run_network and bngsim, reading the same network file, pick different branches when c
        is 0.2, while with c written as a comparison they agree, and that form exports."""
        import shutil
        import subprocess
        pytest.importorskip('petab')
        bngsim = pytest.importorskip('bngsim')
        functions = '  fi() = if(kD - 0.5, kA, kB)\n  fc() = if(kD - 0.5 != 0, kA, kB)'
        (tmp_path / 'm.bngl').write_text(BNG_ORACLE_MODEL.format(functions=functions)
                                         + BNG_ORACLE_ACTIONS)
        proc = subprocess.run([shutil.which('BNG2.pl'), 'm.bngl'], check=False, cwd=tmp_path,
                              capture_output=True, text=True, timeout=300)
        gdat = tmp_path / 'm.gdat'
        assert gdat.exists(), f'BNG2.pl did not simulate:\n{proc.stdout}\n{proc.stderr}'
        lines = gdat.read_text().splitlines()
        run_network = dict(zip(lines[0].lstrip('#').split(), map(float, lines[1].split())))
        result = bngsim.Simulator(bngsim.Model.from_net(str(tmp_path / 'm.net')),
                                  method='ode').run(t_span=(0.0, 1.0), n_points=2)
        by_bngsim = dict(zip(result.expression_names, np.asarray(result.expressions)[0]))
        assert (run_network['fi'], by_bngsim['fi']) == (3.0, 2.0)   # kB, kA
        assert (run_network['fc'], by_bngsim['fc']) == (2.0, 2.0)
        ent = parse_model(BNG_ORACLE_MODEL.format(functions=functions))
        with pytest.raises(NotImplementedError, match=r'if\(\) condition'):
            bngl_body_to_petab_math(ent.function_bodies['fi'], ent)
        formula = bngl_body_to_petab_math(ent.function_bodies['fc'], ent)
        assert _petab_value(formula, BNG_ORACLE_PARAMS) == 2.0

    @pytest.mark.bionetgen
    def test_bng_oracle_table_matches_a_live_bng2pl(self, tmp_path):
        """Re-derive BNG_ORACLE and BNG_ORACLE_SIMULATOR_ONLY from BNG2.pl itself, and check
        every emitted formula against the live .gdat, so the table cannot quietly rot."""
        import shutil
        import subprocess
        pytest.importorskip('petab')
        (tmp_path / 'oracle.bngl').write_text(_oracle_model_text() + BNG_ORACLE_ACTIONS)
        proc = subprocess.run([shutil.which('BNG2.pl'), 'oracle.bngl'], check=False,
                              cwd=tmp_path, capture_output=True, text=True, timeout=300)
        gdat = tmp_path / 'oracle.gdat'
        assert gdat.exists(), f'BNG2.pl did not simulate:\n{proc.stdout}\n{proc.stderr}'
        lines = gdat.read_text().splitlines()
        header = lines[0].lstrip('#').split()
        live = [dict(zip(header, map(float, ln.split()))) for ln in lines[1:]]
        assert [row['obsA'] for row in live] == pytest.approx(BNG_ORACLE_OBS_A, rel=1e-12)
        ent = parse_model(_oracle_model_text())
        for name, body, v0, v1 in BNG_ORACLE + BNG_ORACLE_SIMULATOR_ONLY:
            assert [row[name] for row in live] == pytest.approx([v0, v1], rel=1e-12), name
        for name, body, _v0, _v1 in BNG_ORACLE:
            formula = bngl_body_to_petab_math(body, ent, function_name=name)
            for row in live:
                values = dict(BNG_ORACLE_PARAMS, **row)
                assert _petab_value(formula, values) == pytest.approx(
                    row[name], rel=1e-10, abs=1e-12), (name, formula)


# The issue's reproduction model (#908), verbatim.
PREC_MODEL = """\
begin model
begin parameters
  kA 2
  kB 3
  kC 2
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 10
end seed species
begin observables
  Molecules obsA A()
end observables
begin functions
  f1() = -kC^2 + obsA
  f2() = kA^kB^kC + obsA
  f3() = obsA*exp(-(kC/3)^2)
end functions
begin reaction rules
  A() -> 0 0.1
end reaction rules
end model
"""


# ---------------------------------------------------------------------------
# 2. Syntactic round trip (fast tier) -- export -> import -> re-export (ADR-0036)
# ---------------------------------------------------------------------------

def _write_crafted_src(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'crafted.bngl').write_text(CRAFTED_MODEL)
    (src / 'meas.exp').write_text(CRAFTED_EXP)
    (src / 'job.conf').write_text(CRAFTED_CONF)
    return src


def _func_observable_formula(petab_dir):
    """The inlined/measurement function row's observableFormula (the single func_ row)."""
    import csv
    with open(petab_dir / 'observables.tsv') as fh:
        rows = [r for r in csv.DictReader(fh, delimiter='\t')
                if r['observableId'].startswith('func_')]
    assert len(rows) == 1, rows
    return rows[0]['observableFormula']


def _imported_measurement_formula(imported_dir):
    """The ``observable: <id>, formula: <expr>`` measurement-model formula in the conf."""
    from pybnf.parse import ploop
    conf = ploop((imported_dir / 'imported.conf').read_text().splitlines(keepends=True))
    meas = {k[1]: v for k, v in conf.items()
            if isinstance(k, tuple) and k[0] == 'measurement'}
    assert len(meas) == 1, meas
    return next(iter(meas.values()))


class TestSyntacticRoundTrip:

    def test_expression_formula_round_trips_export_import_reexport(self, tmp_path):
        pytest.importorskip('petab')
        src = _write_crafted_src(tmp_path)
        p1, imported, p2 = tmp_path / 'p1', tmp_path / 'imp', tmp_path / 'p2'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)
        export_job(imported / 'imported.conf', p2, inline_functions=True)

        # The observableFormula survives export -> import(measurement model) -> re-export,
        # equal up to sympy normalization (the importer carries it verbatim; the exporter
        # re-emits it, so this is byte-stable, but graded structurally to be safe).
        assert _sympy_equal(_func_observable_formula(p1), _func_observable_formula(p2))

    def test_import_carries_a_measurement_model_not_a_synthesized_function(self, tmp_path):
        # ADR-0036: no begin-functions synthesis. The imported model is carried verbatim
        # (original `pRel`, no synthesized `func_pRel`), and the measurement model lives in
        # the conf as an `observable: func_pRel, formula:` line whose formula denotes the
        # same function as the original pRel inlined.
        pytest.importorskip('petab')
        src = _write_crafted_src(tmp_path)
        p1, imported = tmp_path / 'p1', tmp_path / 'imp'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)

        ent = parse_model((imported / 'crafted.bngl').read_text())
        assert 'pRel' in ent.function_bodies          # original model carried verbatim
        assert 'func_pRel' not in ent.function_bodies  # NO synthesis into the model
        orig = bngl_body_to_petab_math(ent.function_bodies['pRel'], ent)
        assert _sympy_equal(orig, _imported_measurement_formula(imported))

    def test_imported_expression_problem_passes_petab_validation(self, tmp_path):
        # The external oracle: the re-exported problem (verbatim model + the measurement
        # model's formula) loads through petab's BnglModel and passes every default
        # validation task (so the emitted problem is genuinely valid PEtab, not merely
        # self-consistent).
        pytest.importorskip('petab.v2')
        from petab.v2 import Problem
        from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

        src = _write_crafted_src(tmp_path)
        p1, imported, p2 = tmp_path / 'p1', tmp_path / 'imp', tmp_path / 'p2'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)
        export_job(imported / 'imported.conf', p2, inline_functions=True)

        problem = Problem.from_yaml(str(p2 / 'problem.yaml'))
        assert type(problem.model).__name__ == 'BnglModel'
        errors = [(type(t).__name__, t.run(problem).message)
                  for t in default_validation_tasks
                  if t.run(problem) is not None
                  and getattr(t.run(problem), 'level', None) ==
                  ValidationIssueSeverity.ERROR]
        assert errors == []


# ---------------------------------------------------------------------------
# 3. Semantic round trip (-m recovery, bngsim) -- the strongest oracle
# ---------------------------------------------------------------------------

@pytest.mark.recovery
@pytest.mark.bngsim
class TestSemanticRoundTrip:

    def test_measurement_layer_reproduces_the_original_function_trace(self, tmp_path):
        """Simulate the imported job through the real bngsim backend and apply the
        measurement layer: the layer's computed ``func_pRel`` column must match the original
        BNGL function ``pRel`` the verbatim model still carries, cell-for-cell. A
        self-consistent-but-wrong translation (the failure a syntactic oracle misses) would
        diverge here -- and this exercises the real config -> layer wiring, not a stub."""
        pytest.importorskip('petab')
        import os

        import numpy as np

        from pybnf import config as config_mod
        from pybnf.parse import ploop
        from pybnf.pset import PSet
        from .recovery_harness import build, require_bng2pl
        require_bng2pl()

        src = _write_crafted_src(tmp_path)
        p1, imported = tmp_path / 'p1', tmp_path / 'imp'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)

        # Build a real bngsim config from the imported conf (its `observable: func_pRel,
        # formula:` line builds the measurement layer) and run the verbatim model.
        conf_text = (imported / 'imported.conf').read_text() + '\nbngl_backend = bngsim\n'
        home = os.getcwd()
        os.chdir(imported)
        try:
            conf = config_mod.Configuration(ploop(conf_text.splitlines(keepends=True)))
            assert conf.obj.measurement and len(conf.obj.measurement) == 1
            alg = build(conf, 'de')
            values = {v.name: 2.0 for v in alg.variables}    # nominal kA/kB/kC
            pset = PSet([v.set_value(values[v.name]) for v in alg.variables])
            model = alg.model_list[0].copy_with_param_set(pset)
            os.makedirs(alg.sim_dir, exist_ok=True)
            ds = model.execute(alg.sim_dir, 'meas', 0)
        finally:
            os.chdir(home)

        # Apply the measurement layer post-simulation, exactly as the objective does.
        conf.obj.measurement.apply({model.name: ds}, values)
        data = ds[next(iter(ds))]
        assert 'pRel' in data.cols and 'func_pRel' in data.cols
        np.testing.assert_allclose(data['pRel'], data['func_pRel'], rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Placeholder substitution (ADR-0044): the constant-per-observable reduction primitive
# ---------------------------------------------------------------------------

class TestPlaceholderSubstitution:

    def test_id_token_becomes_a_free_symbol(self):
        # A constant-per-observable observableParameter is substituted by its parameter id,
        # which stays a free symbol (it resolves from the PSet at eval time, ADR-0044).
        pytest.importorskip('petab')
        from pybnf.petab.formula import substitute_placeholders
        out = substitute_placeholders('observableParameter1_obs * x',
                                      {'observableParameter1_obs': 'scaling'})
        assert 'observableParameter' not in out
        assert _sympy_equal(out, 'scaling * x')

    def test_numeric_token_inlines_as_a_constant(self):
        pytest.importorskip('petab')
        from pybnf.petab.formula import substitute_placeholders
        out = substitute_placeholders('0.1 + 0.05*noiseParameter1_obs',
                                      {'noiseParameter1_obs': '4'})
        assert 'noiseParameter' not in out
        assert _sympy_equal(out, '0.1 + 0.05*4')        # == 0.3

    def test_no_substitution_returns_verbatim_without_petab(self, monkeypatch):
        # An empty substitution map is the bare-name / no-placeholder common case: the formula
        # is returned byte-verbatim and petab is never imported (the dependency-free guarantee).
        import builtins
        from pybnf.petab.formula import substitute_placeholders
        real_import = builtins.__import__

        def no_petab(name, *a, **k):
            if name.startswith('petab'):
                raise AssertionError('petab must not be imported for an empty substitution')
            return real_import(name, *a, **k)
        monkeypatch.setattr(builtins, '__import__', no_petab)
        assert substitute_placeholders('x', {}) == 'x'

    def test_unmatched_placeholder_is_left_in_place(self):
        # A placeholder not in the map is left untouched (the caller validates it downstream:
        # the importer raises the deferred-frontier error on a surviving placeholder).
        pytest.importorskip('petab')
        from pybnf.petab.formula import substitute_placeholders
        out = substitute_placeholders('observableParameter1_obs * observableParameter2_obs',
                                      {'observableParameter1_obs': 'scaling'})
        assert 'observableParameter2_obs' in out

    def test_formula_free_symbols_lists_sorted_names(self):
        pytest.importorskip('petab')
        from pybnf.petab.formula import formula_free_symbols
        assert formula_free_symbols('0.1 + 0.05*slope + base') == ['base', 'slope']
        assert formula_free_symbols('0.5') == []        # a pure constant has no free symbols


class TestDerivedSymbolInlining:
    """Inlining a derived SBML entity into a measurement formula (#465 for an assignment rule,
    #795 for a parameter an initialAssignment derives)."""

    @staticmethod
    def _derived(**kwargs):
        from pybnf.petab._sbml import DerivedSymbol
        return {name: DerivedSymbol('initial_assignment', expr, None)
                for name, expr in kwargs.items()}

    def test_a_derived_parameter_inlines_to_its_definition(self):
        pytest.importorskip('petab')
        from pybnf.petab.formula import inline_derived_symbols
        # Bertozzi's shape: the model file gives beta_N no value of its own, so scoring an
        # observable over it used to use the placeholder attribute.
        out = inline_derived_symbols('beta_N * I',
                                     self._derived(beta_N='(R0_ * gamma_) / N_'),
                                     observable_id='rate')
        assert 'beta_N' not in out
        assert {'R0_', 'gamma_', 'N_', 'I'} <= set(re.findall(r'[A-Za-z_]\w*', out))

    def test_an_alias_inlines(self):
        pytest.importorskip('petab')
        from pybnf.petab.formula import inline_derived_symbols
        # Laske's shape: 27 parameters of the form ModelValue_82 = D_rib.
        assert inline_derived_symbols('ModelValue_82',
                                      self._derived(ModelValue_82='D_rib')) == 'D_rib'

    def test_a_chain_resolves_through(self):
        pytest.importorskip('petab')
        from pybnf.petab.formula import inline_derived_symbols
        out = inline_derived_symbols('chain', self._derived(chain='stale * 2', stale='k + 1'))
        assert 'stale' not in out and 'chain' not in out
        assert 'k' in out

    def test_a_formula_naming_nothing_derived_is_returned_verbatim(self):
        pytest.importorskip('petab')
        from pybnf.petab.formula import inline_derived_symbols
        assert inline_derived_symbols('A + B', self._derived(x='k + 1')) == 'A + B'

    def test_a_non_inlinable_initial_assignment_names_the_real_cause(self):
        pytest.importorskip('petab')
        from pybnf.petab._sbml import DerivedSymbol, _r_time_varying
        from pybnf.petab.formula import inline_derived_symbols
        from pybnf.printing import PybnfError
        derived = {'over_species': DerivedSymbol('initial_assignment', None,
                                                 _r_time_varying(['A']))}
        with pytest.raises(PybnfError) as excinfo:
            inline_derived_symbols('over_species * 2', derived, observable_id='rate')
        message = str(excinfo.value)
        assert "Measurement model 'rate'" in message
        assert 'gives no value of its own' in message
        assert "'A'" in message and 'changes during the simulation' in message
        assert '#795' in message

    def test_an_untranslatable_assignment_rule_still_says_assignment_rule(self):
        pytest.importorskip('petab')
        from pybnf.petab._sbml import DerivedSymbol, _R_RULE_UNTRANSLATABLE
        from pybnf.petab.formula import inline_derived_symbols
        from pybnf.printing import PybnfError
        derived = {'ruled': DerivedSymbol('assignment_rule', None, _R_RULE_UNTRANSLATABLE)}
        with pytest.raises(PybnfError, match='assignment rule'):
            inline_derived_symbols('ruled', derived, observable_id='rate')

    def test_a_cyclic_definition_names_the_construct_and_the_chain(self):
        pytest.importorskip('petab')
        from pybnf.petab.formula import inline_derived_symbols
        from pybnf.printing import PybnfError
        with pytest.raises(PybnfError) as excinfo:
            inline_derived_symbols('a', self._derived(a='b', b='a'), observable_id='rate')
        message = str(excinfo.value)
        assert 'initial assignment' in message
        assert 'a -> b -> a' in message
